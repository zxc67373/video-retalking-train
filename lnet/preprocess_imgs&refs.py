import os 
os.environ['CUDA_VISIBLE_DEVICES']='2'
os.environ["NCCL_P2P_DISABLE"]='true'
import numpy as np
import cv2, os, sys, subprocess, platform, torch
from tqdm import tqdm
from PIL import Image
from scipy.io import loadmat
import random
from moviepy.editor import VideoFileClip
import argparse

sys.path.insert(0, 'third_part')
sys.path.insert(0, 'third_part/GPEN')
sys.path.insert(0, 'third_part/GFPGAN')

# 3dmm extraction
from third_part.face3d.util.preprocess import align_img
from third_part.face3d.util.load_mats import load_lm3d
from third_part.face3d.extract_kp_videos import KeypointExtractor
# face enhancement
from third_part.GPEN.gpen_face_enhancer import FaceEnhancement
# expression control

from utils import audio
from utils.ffhq_preprocess import Croper
from utils.alignment_stit import crop_faces, calc_alignment_coefficients, paste_image
from utils.inference_utils import Laplacian_Pyramid_Blending_with_mask, face_detect, load_model, split_coeff, \
                                  trans_image, transform_semantic, find_crop_norm_ratio, load_face3d_net, exp_aus_dict
import warnings
warnings.filterwarnings("ignore")



def main(args):    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[Info] Using {} for inference.'.format(device))
    os.makedirs(os.path.join('temp', args.tmp_dir), exist_ok=True)

    base_name = args.face.split('/')[-1]
    if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        args.static = True
    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')
    elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps
    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        full_frames = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]
            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)

    print ("[Step 0] Number of frames available for inference: "+str(len(full_frames)))
    # face detection & cropping, cropping the first frame as the style of FFHQ
    croper = Croper('checkpoints/shape_predictor_68_face_landmarks.dat')
    full_frames_RGB = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in full_frames]
    full_frames_RGB, crop, quad = croper.crop(full_frames_RGB, xsize=512) # 裁剪后的图像序列、裁剪坐标和四个坐标值

    # 根据四个坐标做缩人脸缩放
    clx, cly, crx, cry = crop
    lx, ly, rx, ry = quad
    lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
    oy1, oy2, ox1, ox2 = cly+ly, min(cly+ry, full_frames[0].shape[0]), clx+lx, min(clx+rx, full_frames[0].shape[1])
    # original_size = (ox2 - ox1, oy2 - oy1)
    frames_pil = [Image.fromarray(cv2.resize(frame,(256,256))) for frame in full_frames_RGB]

    # 获取人脸特征点
    if not os.path.isfile('temp/'+base_name+'_landmarks.txt') or args.re_preprocess:
        print('[Step 1] Landmarks Extraction in Video.')
        kp_extractor = KeypointExtractor()
        lm = kp_extractor.extract_keypoint(frames_pil, './temp/'+base_name+'_landmarks.txt')
    else:
        print('[Step 1] Using saved landmarks.')
        lm = np.loadtxt('temp/'+base_name+'_landmarks.txt').astype(np.float32)
        lm = lm.reshape([len(full_frames), -1, 2])
       
    if not os.path.isfile('temp/'+base_name+'_coeffs.npy') or args.exp_img is not None or args.re_preprocess:
        net_recon = load_face3d_net(args.face3d_net_path, device)
        lm3d_std = load_lm3d('checkpoints/BFM')

        video_coeffs = []
        for idx in tqdm(range(len(frames_pil)), desc="[Step 2] 3DMM Extraction In Video:"):
            frame = frames_pil[idx]
            W, H = frame.size
            lm_idx = lm[idx].reshape([-1, 2])
            if np.mean(lm_idx) == -1:
                lm_idx = (lm3d_std[:, :2]+1) / 2.
                lm_idx = np.concatenate([lm_idx[:, :1] * W, lm_idx[:, 1:2] * H], 1)
            else:
                lm_idx[:, -1] = H - 1 - lm_idx[:, -1]

            trans_params, im_idx, lm_idx, _ = align_img(frame, lm_idx, lm3d_std)
            trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
            im_idx_tensor = torch.tensor(np.array(im_idx)/255., dtype=torch.float32).permute(2, 0, 1).to(device).unsqueeze(0) 
            with torch.no_grad():
                coeffs = split_coeff(net_recon(im_idx_tensor))

            pred_coeff = {key:coeffs[key].cpu().numpy() for key in coeffs}
            pred_coeff = np.concatenate([pred_coeff['id'], pred_coeff['exp'], pred_coeff['tex'], pred_coeff['angle'],\
                                         pred_coeff['gamma'], pred_coeff['trans'], trans_params[None]], 1)
            video_coeffs.append(pred_coeff)
        semantic_npy = np.array(video_coeffs)[:,0]
        np.save('temp/'+base_name+'_coeffs.npy', semantic_npy)
    else:
        print('[Step 2] Using saved coeffs.')
        semantic_npy = np.load('temp/'+base_name+'_coeffs.npy').astype(np.float32)

    # generate the 3dmm coeff from a single image
    if args.exp_img is not None and ('.png' in args.exp_img or '.jpg' in args.exp_img):
        print('extract the exp from',args.exp_img)
        exp_pil = Image.open(args.exp_img).convert('RGB')
        lm3d_std = load_lm3d('third_part/face3d/BFM')
        
        W, H = exp_pil.size
        kp_extractor = KeypointExtractor()
        lm_exp = kp_extractor.extract_keypoint([exp_pil], 'temp/'+base_name+'_temp.txt')[0]
        if np.mean(lm_exp) == -1:
            lm_exp = (lm3d_std[:, :2] + 1) / 2.
            lm_exp = np.concatenate(
                [lm_exp[:, :1] * W, lm_exp[:, 1:2] * H], 1)
        else:
            lm_exp[:, -1] = H - 1 - lm_exp[:, -1]

        trans_params, im_exp, lm_exp, _ = align_img(exp_pil, lm_exp, lm3d_std)
        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
        im_exp_tensor = torch.tensor(np.array(im_exp)/255., dtype=torch.float32).permute(2, 0, 1).to(device).unsqueeze(0)
        with torch.no_grad():
            expression = split_coeff(net_recon(im_exp_tensor))['exp'][0]
        del net_recon
    elif args.exp_img == 'smile':
        expression = torch.tensor(loadmat('checkpoints/expression.mat')['expression_mouth'])[0]
    else:
        print('using expression center')
        expression = torch.tensor(loadmat('checkpoints/expression.mat')['expression_center'])[0]

    # load DNet, model(LNet and ENet)
    D_Net,_ = load_model(args, device)

    if not os.path.isfile('temp/'+base_name+'_stablized.npy') or args.re_preprocess:
        imgs = []
        for idx in tqdm(range(len(frames_pil)), desc="[Step 3] Stabilize the expression In Video:"):
            if args.one_shot:
                source_img = trans_image(frames_pil[0]).unsqueeze(0).to(device)
                semantic_source_numpy = semantic_npy[0:1]
            else:
                source_img = trans_image(frames_pil[idx]).unsqueeze(0).to(device)
                semantic_source_numpy = semantic_npy[idx:idx+1]
            ratio = find_crop_norm_ratio(semantic_source_numpy, semantic_npy)
            coeff = transform_semantic(semantic_npy, idx, ratio).unsqueeze(0).to(device)
        
            # hacking the new expression
            coeff[:, :64, :] = expression[None, :64, None].to(device) 
            with torch.no_grad():
                output = D_Net(source_img, coeff)
            img_stablized = np.uint8((output['fake_image'].squeeze(0).permute(1,2,0).cpu().clamp_(-1, 1).numpy() + 1 )/2. * 255)
            imgs.append(cv2.cvtColor(img_stablized,cv2.COLOR_RGB2BGR)) 
        np.save('temp/'+base_name+'_stablized.npy',imgs)
        del D_Net
    else:
        print('[Step 3] Using saved stabilized video.')
        imgs = np.load('temp/'+base_name+'_stablized.npy')
    torch.cuda.empty_cache()

    if not args.audio.endswith('.wav'):
        command = 'ffmpeg -loglevel error -y -i {} -strict -2 {}'.format(args.audio, 'temp/{}/temp.wav'.format(args.tmp_dir))
        subprocess.call(command, shell=True)
        args.audio = 'temp/{}/temp.wav'.format(args.tmp_dir)
    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_step_size, mel_idx_multiplier, i, mel_chunks = 16, 80./fps, 0, []
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    print("[Step 4] Load audio; Length of mel chunks: {}".format(len(mel_chunks)))
    imgs = imgs[:len(mel_chunks)]
    full_frames = full_frames[:len(mel_chunks)]  
    lm = lm[:len(mel_chunks)]
    
    imgs_enhanced = []
    enhancer = FaceEnhancement(base_dir='checkpoints', size=512, model='GPEN-BFR-512', use_sr=False, \
                            sr_model='rrdb_realesrnet_psnr', channel_multiplier=2, narrow=1, device=device)

    for idx in tqdm(range(len(imgs)), desc='[Step 5] Reference Enhancement'):
        img = imgs[idx]
        pred, _, _ = enhancer.process(img, img, face_enhance=True, possion_blending=False)
        imgs_enhanced.append(pred)
    datagen(args, imgs_enhanced.copy(), mel_chunks, full_frames, None, (oy1,oy2,ox1,ox2))



# frames:256x256, full_frames: original size
def datagen(args, frames, mels, full_frames, frames_pil, cox):
    img_batch, mel_batch, frame_batch, coords_batch, ref_batch, full_frame_batch = [], [], [], [], [], []
    base_name = args.face.split('/')[-1]
    refs = []
    image_size = 256 

    # original frames
    kp_extractor = KeypointExtractor()
    fr_pil = [Image.fromarray(frame) for frame in frames]
    lms = kp_extractor.extract_keypoint(fr_pil, 'temp/'+base_name+'x12_landmarks.txt')
    frames_pil = [ (lm, frame) for frame,lm in zip(fr_pil, lms)] # frames is the croped version of modified face
    crops, orig_images, quads  = crop_faces(image_size, frames_pil, scale=1.0, use_fa=True)
    inverse_transforms = [calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]]) for quad in quads]
    del kp_extractor.detector

    oy1,oy2,ox1,ox2 = cox
    face_det_results = face_detect(full_frames, args, jaw_correction=True)

    for inverse_transform, crop, full_frame, face_det in zip(inverse_transforms, crops, full_frames, face_det_results):
        imc_pil = paste_image(inverse_transform, crop, Image.fromarray(
            cv2.resize(full_frame[int(oy1):int(oy2), int(ox1):int(ox2)], (256, 256))))

        ff = full_frame.copy()
        ff[int(oy1):int(oy2), int(ox1):int(ox2)] = cv2.resize(np.array(imc_pil.convert('RGB')), (ox2 - ox1, oy2 - oy1))
        oface, coords = face_det
        y1, y2, x1, x2 = coords
        refs.append(ff[y1: y2, x1:x2])

    
    for i, m in enumerate(mels):
        if len(refs) != len(mels):
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face = refs[idx]
        oface, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size)) # 参考人脸图片
        oface = cv2.resize(oface, (args.img_size, args.img_size)) # 原始人脸图片

        img_name = args.face
        img_name = img_name.replace('cds','cds_refs').replace('.MP4','').replace('.mp4','')
        # img_name = 'temp/refs' # 处理单个文件，放到temp
        if not os.path.exists(img_name):
            os.makedirs(img_name)
        cv2.imwrite(f'{img_name}/{i}.jpg',face)

        img_name = img_name.replace('cds_refs','cds_imgs')
        # img_name = 'temp/imgs'
        if not os.path.exists(img_name):
            os.makedirs(img_name)
            os.system(f'mv {args.audio} {img_name}/audio.wav')
            np.save(f'{img_name}/mel_chunks.npy',mels)
        cv2.imwrite(f'{img_name}/{i}.jpg',oface)


def options():
    parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

    parser.add_argument('--DNet_path', type=str, default='checkpoints/DNet.pt')
    parser.add_argument('--LNet_path', type=str, default='checkpoints/LNet.pth')
    parser.add_argument('--ENet_path', type=str, default='checkpoints/ENet.pth')
    parser.add_argument('--face3d_net_path', type=str, default='checkpoints/face3d_pretrain_epoch_20.pth')                      
    parser.add_argument('--face', type=str, help='Filepath of video/image that contains faces to use',  default='')
    parser.add_argument('--audio', type=str, help='Filepath of video/audio file to use as raw audio source', default='')
    parser.add_argument('--exp_img', type=str, help='Expression template. neutral, smile or image path', default='neutral')

    parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', default=25., required=False)
    parser.add_argument('--pads', nargs='+', type=int, default=[0, 20, 0, 0], help='Padding (top, bottom, left, right). Please adjust to include chin at least')
    parser.add_argument('--face_det_batch_size', type=int, help='Batch size for face detection', default=4)
    parser.add_argument('--LNet_batch_size', type=int, help='Batch size for LNet', default=16)
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
                        help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
                        'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')
    parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
                        help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                        'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')
    parser.add_argument('--nosmooth', default=False, action='store_true', help='Prevent smoothing face detections over a short temporal window')
    parser.add_argument('--static', default=False, action='store_true')

    parser.add_argument('--up_face', default='original')
    parser.add_argument('--one_shot', action='store_true')
    parser.add_argument('--without_rl1', default=False, action='store_true', help='Do not use the relative l1')
    parser.add_argument('--tmp_dir', type=str, default='temp', help='Folder to save tmp results')
    parser.add_argument('--re_preprocess', action='store_true',default=False)
    
    args = parser.parse_args()
    return args

def get_audio(video_path,audio_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path)

if __name__ == '__main__':
    args = options()
    for dir in os.listdir('/datadisk1/zhouxc/CY/Lips/video_retalk/lnet/_data_/cds/'):
        listdir_path = f'/datadisk1/zhouxc/CY/Lips/video_retalk/lnet/_data_/cds/{dir}'
        for i in os.listdir(listdir_path):
            args.face = os.path.join(listdir_path,i)

            if 'nianzi' not in args.face: continue 

            if not args.face.endswith(('.mp4','.MP4')):continue
            args.audio = args.face.replace('.MP4','.wav').replace('.mp4','.wav')
            print(args.face,args.audio)
            get_audio(args.face,args.audio)
            try:
                main(args)
            except:
                print(args.face)
                img_name = args.face
                img_name = img_name.replace('cds','cds_refs').replace('.MP4','')
                os.system(f'rm -rf {img_name}')
                img_name = img_name.replace('cds_refs','cds_imgs')
                os.system(f'rm -rf {img_name}')
        #     break
        # break

# from multiprocessing import Pool
# import multiprocessing
# multiprocessing.set_start_method('spawn', force=True)

# def process_file(file_path):
#     args = options()
#     args.face = file_path
#     if not args.face.endswith(('.mp4','.MP4')): return
#     args.audio = args.face.replace('.MP4','.wav').replace('.mp4','.wav')
#     get_audio(args.face,args.audio)
#     # main(args)
#     try:
#         main(args)
#     except:
#         print(args.face)
#         img_name = args.face
#         img_name = img_name.replace('cds','cds_refs').replace('.MP4','')
#         os.system(f'rm -rf {img_name}')
#         img_name = img_name.replace('cds_refs','cds_imgs')
#         os.system(f'rm -rf {img_name}')

# if __name__ == '__main__':
#     listdir_path = '/datadisk1/zhouxc/CY/Lips/video_retalk/lnet/_data_/cds/'
#     all_files = []
#     for dir in os.listdir(listdir_path):
#         dir_path = os.path.join(listdir_path, dir)
#         for i in os.listdir(dir_path):
#             file_path = os.path.join(dir_path, i)
#             all_files.append(file_path)

#     with Pool(processes=3) as pool:
#         pool.map(process_file, all_files)
