import os
import sys
import torch
import torch.nn as nn
import numpy as np
# import lpips
from tqdm import tqdm
from torchvision.utils import save_image
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler



def cosine_loss(a, v, y):
    logloss = nn.BCELoss()
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)
    return loss


def get_sync_loss(syncnet, mel, g, device):
    # g = g[:, :, :, g.size(3)//2:]
    # g = g.repeat_interleave(5, dim=1)  # 在第二个维度（dim=1）上重复每个元素5次
    # g = g[:, :, g.size(3)//2:, g.size(3)//2:]
    # B, 3 * T, H//2, W
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)


def train_one_epoch(args, lnet, disc, syncnet, optim_lnet, optim_disc,
                    data_loader, device, epoch):
    lnet.train()
    disc.train()
    syncnet.eval()

    BCE = nn.BCELoss()
    # L1 = lpips.LPIPS(net='alex').to(device)
    L1 = nn.SmoothL1Loss()

    scaler = GradScaler()
    loss_dict = {'lnet_loss': 0., 'disc_loss': 0., 'sync_loss': 0., 'pix_loss': 0.}

    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, mask_images, dnet_images, mel_spectrogram = data

        images = images.float().to(device)
        mask_images = mask_images.float().to(device)
        dnet_images = dnet_images.float().to(device)
        mel_spectrogram = mel_spectrogram.float().to(device)

        comb_images = torch.cat([mask_images, dnet_images], dim=1)

        # 训练判别器
        disc.zero_grad()
        # with autocast():
        # 生成假图
        fakes = lnet(mel_spectrogram, comb_images)

        real_disc = disc(images)
        fake_disc = disc(fakes.detach())

        real_disc_loss = BCE(real_disc, torch.ones_like(real_disc))
        fake_disc_loss = BCE(fake_disc, torch.zeros_like(fake_disc))
        disc_loss = (real_disc_loss + fake_disc_loss) / 2

        # scaler.scale(disc_loss).backward()
        # scaler.step(optim_disc)
        # scaler.update()
        disc_loss.backward()
        optim_disc.step()



        lnet.zero_grad()
        # with autocast():
        sync_loss = get_sync_loss(syncnet=syncnet,
                                    mel=mel_spectrogram,
                                    g=fakes,
                                    device=device)

        fake_disc_lnet = disc(fakes)

        fake_disc_lnet_loss = BCE(fake_disc_lnet,
                                    torch.ones_like(fake_disc_lnet))
        pix_loss = L1(fakes, images).mean()

        lnet_loss = fake_disc_lnet_loss + sync_loss + pix_loss
        lnet_loss.backward()
        optim_lnet.step()
        # scaler.scale(lnet_loss).backward()
        # scaler.step(optim_lnet)
        # scaler.update()

        with torch.no_grad():
            loss_dict['lnet_loss'] += lnet_loss.item()
            loss_dict['disc_loss'] += disc_loss.item()
            loss_dict['sync_loss'] += sync_loss.item()
            loss_dict['pix_loss'] += pix_loss.item()


            data_loader.desc = "$$ TRA-e{},[lnet:{:.5f},disc:{:.5f},sync:{:.5f},pix:{:5f}] AVG:[lnet:{:.5f},disc:{:.5f},sync:{:.5f},pix:{:5f}]".format(
                epoch + 1,
                lnet_loss.item(),
                disc_loss.item(),
                sync_loss.item(),
                pix_loss.item(),
                loss_dict['lnet_loss'] / (step + 1),
                loss_dict['disc_loss'] / (step + 1),
                loss_dict['sync_loss'] / (step + 1),
                loss_dict['pix_loss'] / (step + 1),
            )

    loss_dict = {k: v / (step + 1) for k, v in loss_dict.items()}
    return loss_dict


@torch.no_grad()
def eval(args, lnet, disc, syncnet,
                    data_loader, device, epoch):
    lnet.eval()
    disc.eval()
    syncnet.eval()

    BCE = nn.BCELoss()
    # L1 = lpips.LPIPS(net='alex').to(device)
    L1 = nn.SmoothL1Loss()

    loss_dict = {'lnet_loss': 0., 'disc_loss': 0., 'sync_loss': 0.,'pix_loss': 0.}

    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, mask_images, dnet_images, mel_spectrogram = data

        images = images.float().to(device)
        mask_images = mask_images.float().to(device)
        dnet_images = dnet_images.float().to(device)
        mel_spectrogram = mel_spectrogram.float().to(device)

        comb_images = torch.cat([mask_images, dnet_images], dim=1)

        fakes = lnet(mel_spectrogram, comb_images)

        real_disc = disc(images)
        fake_disc = disc(fakes)

        real_disc_loss = BCE(real_disc, torch.ones_like(real_disc))
        fake_disc_loss = BCE(fake_disc, torch.zeros_like(fake_disc))
        disc_loss = (real_disc_loss + fake_disc_loss) / 2

        sync_loss = get_sync_loss(syncnet=syncnet,
                                    mel=mel_spectrogram,
                                    g=fakes,
                                    device=device)

        fake_disc_lnet = disc(fakes)

        fake_disc_lnet_loss = BCE(fake_disc_lnet,
                                    torch.ones_like(fake_disc_lnet))
        pix_loss = L1(fakes, images).mean()

        lnet_loss = fake_disc_lnet_loss + sync_loss + pix_loss


 
        loss_dict['lnet_loss'] += lnet_loss.item()
        loss_dict['disc_loss'] += disc_loss.item()
        loss_dict['sync_loss'] += sync_loss.item()
        loss_dict['pix_loss'] += pix_loss.item()

        data_loader.desc = "$$ VAL-e{},[lnet:{:.5f},disc:{:.5f},sync:{:.5f},pix:{:5f}] AVG:[lnet:{:.5f},disc:{:.5f},sync:{:.5f},pix:{:5f}]".format(
            epoch + 1,
            lnet_loss.item(),
            disc_loss.item(),
            sync_loss.item(),
            pix_loss.item(),
            loss_dict['lnet_loss'] / (step + 1),
            loss_dict['disc_loss'] / (step + 1),
            loss_dict['sync_loss'] / (step + 1),
            loss_dict['pix_loss'] / (step + 1),
        )
    
    loss_dict = {k: v / (step + 1) for k, v in loss_dict.items()}
    if (epoch + 1) % 1 == 0:
        print("Saving images...")
        if args.save_img_path is not None:
            os.makedirs(args.save_img_path, exist_ok=True)
            batch_size = images.size(0)
            save_image(fakes, os.path.join(args.save_img_path, f"fake_epoch_{epoch + 1}.png"), nrow=batch_size // 5)
            save_image(images, os.path.join(args.save_img_path, f"real_epoch_{epoch + 1}.png"), nrow=batch_size // 5)
    return loss_dict