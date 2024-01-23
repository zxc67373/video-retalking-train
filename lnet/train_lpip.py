import os
import cv2
import random
import torch
import torch.nn as nn
import numpy as np
import lpips
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

PRECETUAL = lpips.LPIPS(net='vgg').cuda() # 感知损失
L1 = nn.SmoothL1Loss() # L1损失
BCE = nn.BCELoss()

def cosine_loss(a, v, y):
    logloss = nn.BCELoss()
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)
    return loss


def get_sync_loss(syncnet, mel, g, device):
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)


def train_one_epoch(args, lnet, disc, syncnet, optim_lnet,optim_disc,
                    data_loader, device, epoch,val_loader):
    lnet.train()
    syncnet.eval()

    loss_dict = {'lnet_loss': 0., 'sync_loss': 0., 'pix_loss': 0., 'precetual_loss':0.}

    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, mask_images, dnet_images, mel_spectrogram = data

        images = images.float().to(device)
        mask_images = mask_images.float().to(device)
        dnet_images = dnet_images.float().to(device)
        mel_spectrogram = mel_spectrogram.float().to(device)

        comb_images = torch.cat([mask_images, dnet_images], dim=1)

        # 生成假图
        fakes = lnet(mel_spectrogram, comb_images)

        # debug 使用
        if random.randint(0,5) == 1:
            fake_image = np.array(fakes[0].permute(1, 2, 0).cpu().detach())[:, :, ::-1] * 255
            train_image = np.array(dnet_images[0].permute(1, 2, 0).cpu().detach())[:, :, ::-1] * 255
            train_mask = np.array(mask_images[0].permute(1, 2, 0).cpu().detach())[:, :, ::-1] * 255
            true = np.array(images[0].permute(1, 2, 0).cpu().detach())[:, :, ::-1] * 255
            combined_image = np.hstack((fake_image, train_image, train_mask,true))
            cv2.imwrite('train_res.jpg', combined_image)


        # 训练判别器
        real_disc = disc(images)
        fake_disc = disc(fakes.detach())
        real_disc_loss = BCE(real_disc, torch.ones_like(real_disc))
        fake_disc_loss = BCE(fake_disc, torch.zeros_like(fake_disc))
        disc_loss = (real_disc_loss + fake_disc_loss) / 2
        disc_loss.backward()
        if step%1==0: # 梯度累计
            optim_disc.step()
            optim_disc.zero_grad()

        # 训练生成器
        pix_loss = L1(fakes, images).mean()
        precetual_loss = PRECETUAL(fakes, images).mean()
        # fakes = F.interpolate(fakes, (96,96)) # 当sync和lent输入不一致
        sync_loss = get_sync_loss(syncnet=syncnet,
                                    mel=mel_spectrogram,
                                    g=fakes,
                                    device=device)
        fake_disc_lnet = disc(fakes)
        gan_loss = BCE(fake_disc_lnet,torch.ones_like(fake_disc_lnet))
        lnet_loss =  0.5*sync_loss + pix_loss + precetual_loss + gan_loss*0.
        lnet_loss.backward()
        if step%1==0: # 梯度累计
            optim_lnet.step()
            optim_lnet.zero_grad()


        with torch.no_grad():
            data_loader.desc = "$$ TRA-e{},[lnet:{:.5f},sync:{:.5f},pix:{:5f}] ".format(
                epoch + 1,
                lnet_loss.item(),
                sync_loss.item(),
                pix_loss.item(),
                precetual_loss.item(),
                gan_loss.item()
            )
        if args.save_model and (step % 10000 == 0):
            os.makedirs(args.save_model_path,exist_ok=True)
            val_dict = eval(args=args,
                    lnet=lnet,
                    syncnet=syncnet,
                    data_loader=val_loader,
                    device=device,
                    epoch=epoch)
            torch.save(
                lnet.state_dict(),
                "./{}/epoch{}lnet-loss-{:.2f}.pth".format(
                    args.save_model_path, epoch,
                    (val_dict['lnet_loss'])))
        
    loss_dict = {k: v / (step + 1) for k, v in loss_dict.items()}
    return loss_dict


@torch.no_grad()
def eval(args, lnet, syncnet,
                    data_loader, device, epoch):
    lnet.eval()
    syncnet.eval()

    loss_dict = {'lnet_loss': 0.,  'sync_loss': 0.,'pix_loss': 0.}

    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, mask_images, dnet_images, mel_spectrogram = data

        images = images.float().to(device)
        mask_images = mask_images.float().to(device)
        dnet_images = dnet_images.float().to(device)
        mel_spectrogram = mel_spectrogram.float().to(device)

        comb_images = torch.cat([mask_images, dnet_images], dim=1)

        fakes = lnet(mel_spectrogram, comb_images)

        # debug 使用
        if random.randint(0,5) == 1:
            fake_image = np.array(fakes[0].permute(1, 2, 0).cpu().detach())[:, :, ::-1] * 255
            train_image = np.array(dnet_images[0].permute(1, 2, 0).cpu().detach())[:, :, ::-1] * 255
            train_mask = np.array(mask_images[0].permute(1, 2, 0).cpu().detach())[:, :, ::-1] * 255
            true = np.array(images[0].permute(1, 2, 0).cpu().detach())[:, :, ::-1] * 255
            combined_image = np.hstack((fake_image, train_image, train_mask,true))
            cv2.imwrite('test_res.jpg', combined_image)

        precetual_loss = PRECETUAL(fakes, images).mean()
        pix_loss = L1(fakes, images).mean()


        # fakes = F.interpolate(fakes, (96,96)) #同上

        sync_loss = get_sync_loss(syncnet=syncnet,
                                    mel=mel_spectrogram,
                                    g=fakes,
                                    device=device)
        lnet_loss =   0.3*sync_loss + pix_loss + precetual_loss
 
        loss_dict['lnet_loss'] += lnet_loss.item()
        loss_dict['sync_loss'] += sync_loss.item()
        loss_dict['pix_loss'] += pix_loss.item()

        data_loader.desc = "$$ VAL-e{},[lnet:{:.5f},sync:{:.5f},pix:{:5f}] AVG:[lnet:{:.5f},sync:{:.5f},pix:{:5f}]".format(
            epoch + 1,
            lnet_loss.item(),
            sync_loss.item(),
            pix_loss.item(),
            loss_dict['lnet_loss'] / (step + 1),
            loss_dict['sync_loss'] / (step + 1),
            loss_dict['pix_loss'] / (step + 1),
        )
    
    loss_dict = {k: v / (step + 1) for k, v in loss_dict.items()}
    if (epoch + 1) % 1 == 0:
        print("Saving images...")
        if args.save_img_path is not None:
            os.makedirs(args.save_img_path, exist_ok=True)
            batch_size = images.size(0)
            save_image(fakes, os.path.join(args.save_img_path, f"fake_epoch_{epoch + 1}.png"), nrow=5)
            save_image(images, os.path.join(args.save_img_path, f"real_epoch_{epoch + 1}.png"), nrow=5)
    return loss_dict