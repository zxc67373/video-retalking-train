import os
import random
from time import sleep
import unittest
import numpy as np
import torchaudio
import torchvision.transforms as transforms
from PIL import Image

import torch
from torch.utils.data import Dataset
import os
from utils import audio


def get_mel_segment(mel, mel_step_size, frame_rate=25):
    mel_step_size, mel_idx_multiplier, i, mel_chunks = 16, 80./frame_rate, 0, []
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx:start_idx + mel_step_size])
        i += 1

    return mel_chunks



def get_sequential_frames_and_audio(image_file):
    directory = os.path.dirname(image_file)
    file_name = os.path.basename(image_file)
    # 根据选择的帧的时间戳，从音频文件中提取对应的音频片段并将其转换为Mel频谱
    mel_chunks_file = os.path.join(directory, 'mel_chunks.npy')
    mel_chunks = np.load(mel_chunks_file)

    idx = int(file_name.split('.')[0])
    if idx<len(mel_chunks):
        mel_tensor=torch.FloatTensor(mel_chunks[idx]).unsqueeze(0)
    else:
        print('idx out of range')
        mel_tensor=torch.FloatTensor(mel_chunks[-1]).unsqueeze(0)
        print(mel_chunks_file)

    image = Image.open(image_file)
    return image, mel_tensor

def get_all_image_paths(directory):
    # train
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')) :
                image_paths.append(os.path.join(root, file))
    return image_paths


class SequentialFramesAndAudioDataset(Dataset):

    def __init__(self, fold_path):
        self.fold_path = fold_path
        # self.frame_nums = frame_nums
        # self.frame_rate = frame_rate
        self.image_file_paths = get_all_image_paths(self.fold_path)
        self.transform = transforms.Compose(
        [transforms.Resize((224, 224)), # 放大
         transforms.ToTensor()])  # 预处理代码

    def __len__(self):
        return len(self.image_file_paths)

    def __getitem__(self, idx):
        image_file = self.image_file_paths[idx]
        image, mel = get_sequential_frames_and_audio(image_file)
        image=self.transform(image)
        
        dnet_image_file=image_file.replace('imgs','refs')
        dnet_image_file = '/'.join(image_file.split('/')[:-1])+'/0.jpg'# 
        dnet_image = Image.open(dnet_image_file)
        dnet_image = self.transform(dnet_image)

        mask_image = image.clone()
        img_size = image.shape[2]
        # mask_image[:, img_size // 2:] = 0 
        mask_image[:,img_size//2:,30:img_size-30] = 0 # 修改为使用中间部分mask
        return image, mask_image, dnet_image, mel



if __name__ == '__main__':
    train_dataset = SequentialFramesAndAudioDataset(
        fold_path='_data_/cds_imgs',
       )
    for _, i in enumerate(train_dataset):
        images, mask_images, dnet_images, mel_spectrogram = i
        import torchvision.utils as vutils

        # 将图片合并成网格
        grid = vutils.make_grid([images, mask_images, dnet_images], nrow=3, padding=2, normalize=True)

        # 保存合并后的图片
        vutils.save_image(grid, "merged_images.png")
        print(mask_images.shape)
        print(mel_spectrogram.shape)
        print(images.shape)
        print(dnet_images.shape)
        break
