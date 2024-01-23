import os 
os.environ['CUDA_VISIBLE_DEVICES']='3'
os.environ["NCCL_P2P_DISABLE"]='true'
import sys
import torch
import argparse
import datetime
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import SequentialFramesAndAudioDataset
from models.LNet import LNet
from models.wav2lip import Wav2Lip_disc_qual as DiscNet
from models.syncnet import SyncNet_color as SyncNets
from train import train_one_epoch, eval
from utils.util import Logger, MultiEpochsDataLoader
from torch.utils.data import random_split


def main(args):
    T_max = 25
    norm = True
    nw = 24
    device = torch.device('cuda')
    print('[device]: ', device)

    ### 读取数据 ###
    train_dataset = SequentialFramesAndAudioDataset(fold_path=args.fold_path)
    test_dataset = SequentialFramesAndAudioDataset(fold_path=args.fold_path.replace('cds_imgs','cds_imgs_test'))

    print('train number: {}, val number: {}'.format(len(train_dataset),
                                                    len(test_dataset)))

    train_loader = MultiEpochsDataLoader(train_dataset,
                                         batch_size=args.batch_size,
                                         num_workers=nw,
                                         shuffle=True)
    val_loader = MultiEpochsDataLoader(test_dataset,
                                       batch_size=args.batch_size,
                                       num_workers=nw)

    tb_writer = SummaryWriter('./_save_/runs')

    ### 网络设置 ###
    lnet = LNet().to(device)
    disc = DiscNet().to(device)
    syncnet = SyncNets().to(device)
    for p in syncnet.parameters():
        p.requires_grad = False

    print("Load checkpoint from: {}".format(args.load_syncnet))
    checkpoint_syncnet = torch.load(args.load_syncnet, map_location=device)
    print(syncnet.load_state_dict(checkpoint_syncnet["state_dict"]))

    optim_lnet = optim.AdamW(lnet.parameters(), lr=args.lr, weight_decay=5E-2)
    optim_disc = optim.AdamW(disc.parameters(), lr=args.lr, weight_decay=5E-2)

    scheduler_lnet = optim.lr_scheduler.CosineAnnealingLR(optim_lnet,
                                                          T_max=T_max,
                                                          eta_min=args.lr_min,
                                                          verbose=True)
    scheduler_disc = optim.lr_scheduler.CosineAnnealingLR(optim_disc,
                                                          T_max=T_max,
                                                          eta_min=args.lr_min,
                                                          verbose=True)
    ### 读取权重 ###
    if args.load_model:
        if os.path.isfile(args.load_lnet):
            print("=> loading checkpoint '{}'".format(args.load_lnet))
            checkpoint_lnet = torch.load(args.load_lnet, map_location=device)
            new_s = {}
            for k, v in checkpoint_lnet["state_dict"].items():
                if 'low_res' in k:
                    continue
                else:
                    new_s[k.replace('module.', '')] = v
            print(lnet.load_state_dict(new_s, strict=False))
        if os.path.isfile(args.load_disc):
            print("=> loading checkpoint '{}'".format(args.load_disc))
            print(
                disc.load_state_dict(
                    torch.load(args.load_disc, map_location=device)))

    else:
        print("no lnet model load")

    ### 训练 ###
    if args.mode == 'train':
        for epoch in range(args.epochs):
            train_dict = train_one_epoch(args=args,
                                         lnet=lnet,
                                         disc=disc,
                                         syncnet=syncnet,
                                         optim_lnet=optim_lnet,
                                         optim_disc=optim_disc,
                                         data_loader=train_loader,
                                         device=device,
                                         epoch=epoch)
 
            val_dict = eval(args=args,
                            lnet=lnet,
                            disc=disc,
                            syncnet=syncnet,
                            data_loader=val_loader,
                            device=device,
                            epoch=epoch)

            scheduler_lnet.step()
            scheduler_disc.step()

            tb_writer.add_scalar('train/lnet_loss', train_dict['lnet_loss'],
                                 epoch)
            tb_writer.add_scalar('train/disc_loss', train_dict['disc_loss'],
                                 epoch)
            tb_writer.add_scalar('train/sync_loss', train_dict['sync_loss'],
                                 epoch)
            tb_writer.add_scalar('eval/lnet_loss', val_dict['lnet_loss'],
                                 epoch)
            tb_writer.add_scalar('eval/disc_loss', val_dict['disc_loss'],
                                 epoch)
            tb_writer.add_scalar('eval/sync_loss', val_dict['sync_loss'],
                                 epoch)

            if args.save_model and (epoch + 1) % 5 == 0:
                torch.save(
                    lnet.state_dict(),
                    "./{}/epoch{}lnet-loss-{:.2f}.pth".format(
                        args.save_model_path, epoch,
                        (val_dict['sync_loss'] + val_dict['lnet_loss'])))
                torch.save(
                    disc.state_dict(),
                    "./{}/epoch{}disc-loss-{:.2f}.pth".format(
                        args.save_model_path, epoch, val_dict['disc_loss']))


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(now)
    parser = argparse.ArgumentParser(description='PyTorch face generation')
    # Training parameters
    parser.add_argument('--mode', default='train', help='train or test')
    parser.add_argument('--batch_size',
                        type=int,
                        default=4,
                        metavar='N',
                        help='input batch size for training')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr',
                        type=float,
                        default=1.E-5,
                        metavar='LR',
                        help='learning rate')
    parser.add_argument('--lr_min',
                        type=float,
                        default=1.E-8,
                        metavar='LR_M',
                        help='min learning rate')
    parser.add_argument('--ngpu',
                        type=list,
                        default=[1],
                        help='number of GPUs to use')
    # Save options
    parser.add_argument('--save_img',
                        default=True,
                        help='Choose whether to save image')
    parser.add_argument('--save_model',
                        default=True,
                        help='Choose whether to save model')
    parser.add_argument('--load_model', default=True, help='retrain model')
    # Path
    parser.add_argument('--save_img_path', default='./_save_/train_save/IMG', help='folder to save images')
    parser.add_argument('--save_model_path',
                        default='./_save_/train_save/Weight',
                        help='folder to save models')
    # parser.add_argument('--test_out', default='./_save_/test_save', help='folder to save test')

    parser.add_argument('--fold_path',
                        default='_data_/cds_preproc',
                        help='training image path')

    # Weight
    parser.add_argument(
        '--load_lnet',
        default='/datadisk/huangkf/retalk_retrain/lnet/checkpoints/LNet.pth',
        help='lnet model load path')
    parser.add_argument('--load_disc', default='_save_/train_save/Weight/2023-11-22_09-52-17/epoch14disc-loss-0.70.pth', help='dics model load path')
    parser.add_argument(
        '--load_syncnet',
        default='/datadisk1/zhouxc/CY/Lips/video_retalk/lnet/checkpoints/sync/sync_2024-01-15-14-31-56_0.242255.pth',
        help='sync model load path')

    args = parser.parse_args()
    args.save_img_path = args.save_img_path + '/' + now
    args.save_model_path = args.save_model_path + '/' + now

    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    if not os.path.exists('./_save_/logs'):
        os.makedirs('./_save_/logs')
    sys.stdout = Logger(f'./_save_/logs/log_{now}.log', sys.stdout)
    sys.stderr = Logger(f'./_save_/logs/log_{now}.log', sys.stderr)
    main(args)
