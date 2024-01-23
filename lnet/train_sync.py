import os 
os.environ['CUDA_VISIBLE_DEVICES']='7'
os.environ["NCCL_P2P_DISABLE"]='true'

from os.path import dirname, join, basename, isfile
from matplotlib import pyplot as plt
from tqdm import tqdm

from models.syncnet import SyncNet_color as SyncNet

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np
import datetime

from glob import glob
from datasets_sync import SequentialFramesAndAudioDataset
import os, random, cv2, argparse
from torch.utils.data import random_split
from utils.util import MultiEpochsDataLoader
parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", default='./_data_/cds_imgs')

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', default='./checkpoints/sync', type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: {}'.format(device))
now=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    
    global_step=0
    train_loss_list=[]
    test_loss_list=[]
    for epoch_iter in range(nepochs):
        running_loss = 0.
        prog_bar = tqdm(train_data_loader)
        for step, (x, mel, y) in enumerate(prog_bar):
            model.train()
            optimizer.zero_grad()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            a, v = model(mel, x)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
            loss.backward()
            optimizer.step()

            global_step += 1
            running_loss += loss.item()

            
            prog_bar.set_description('[train epoch {}/{}]: Loss: {}'.format(epoch_iter,nepochs,running_loss / (step + 1)))
        train_loss_list.append(running_loss / (step + 1))
        if (epoch_iter+1) % 1 == 0:
            with torch.no_grad():
                eval_loss=eval_model(test_data_loader, global_step, device, model, checkpoint_dir)
                test_loss_list.append(eval_loss)
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, epoch_iter,eval_loss)
        
    # Plot and save the loss curves
    plt.subplot(2, 1, 1)
    plt.plot(train_loss_list, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(0, len(test_loss_list)*5, 5), test_loss_list, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplots_adjust(hspace=0.5)  # 调整子图之间的垂直间距

    plt.savefig('./loss_sync.png')
    plt.close()
    

def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    # eval_steps = 1400
    # print('Evaluating for {} steps'.format(eval_steps))
    losses = 0.
    bars = tqdm(test_data_loader)
    # while 1:
    for step, (x, mel, y) in enumerate(bars):

        model.eval()

        # Transform data to CUDA device
        x = x.to(device)

        mel = mel.to(device)

        a, v = model(mel, x)
        y = y.to(device)

        loss = cosine_loss(a, v, y)
        losses+=loss.item()
        bars.set_description('[eval]: Loss: {}'.format(losses/(step+1)))
        

        # if step > eval_steps: break

    # averaged_loss = sum(losses) / len(losses)
    # print(averaged_loss)

    return losses/(step+1)

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, eval_loss):
    global now
    checkpoint_path = join(
        checkpoint_dir, "sync_{}_{:5f}.pth".format(now,eval_loss))
    optimizer_state = optimizer.state_dict() 
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
        "eval_loss":eval_loss
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def _load(checkpoint_path):
    if device!="cpu":
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    # global global_step
    # global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    # global_step = checkpoint["global_step"]
    # global_epoch = checkpoint["global_epoch"]

    return model

if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path
    data_path=args.data_root
    batch_size=200
    nw=16

    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)

    # Dataset and Dataloader setup

    full_dataset = SequentialFramesAndAudioDataset(data_path)
    dataset_size = len(full_dataset)
    train_size = int(0.9 * dataset_size)
    test_size = dataset_size - train_size


    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size],torch.Generator().manual_seed(66))
    print('train number: {}, val number: {}'.format(len(train_dataset),
                                                    len(test_dataset)))

    train_data_loader = MultiEpochsDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=nw)

    test_data_loader = MultiEpochsDataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=8)

    # device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = SyncNet().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=0.0001)

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)

    train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=1000,
          nepochs=200)
