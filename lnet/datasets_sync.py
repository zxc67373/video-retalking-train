import os
import random
from time import sleep
import unittest
import torchaudio
import torchvision.transforms as transforms
from PIL import Image

import torch
from torch.utils.data import Dataset
import os
from utils import audio
import numpy as np


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

    audio_file = [
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.endswith('.wav')
    ][0]

    mel_chunks_file = os.path.join(directory, 'mel_chunks.npy')

    if os.path.exists(mel_chunks_file):
        mel_chunks = np.load(mel_chunks_file)
    else:
        wav = audio.load_wav(audio_file, 16000)
        mel = audio.melspectrogram(wav)
        mel_chunks = get_mel_segment(mel, 16, frame_rate=25)
        np.save(mel_chunks_file, mel_chunks)

    if random.choice([True, False]):
        label = torch.ones(1).float()
        idx = int(file_name.split('.')[0]) 
    else:
        label = torch.zeros(1).float()
        idx = random.randint(0, len(mel_chunks) - 1)
        while idx == int(file_name.split('.')[0]):
            idx = random.randint(0, len(mel_chunks) - 1)

    # print(int(file_name.split('.')[0]), idx)
    if idx<len(mel_chunks):
        mel_tensor=torch.FloatTensor(mel_chunks[idx]).unsqueeze(0)
    else:
        print('idx out of range')
        mel_tensor=torch.FloatTensor(mel_chunks[-1]).unsqueeze(0)


    # 读取图像文件并将它们转换为张量
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor()])  # 预处理代码
    # images = [transform(Image.open(f)) for f in selected_files]
    # images = torch.cat(images, dim=0)
    image = transform(Image.open(image_file))
    

    # mel_spectrogram = torch.ones(1,80,16) # TODO: 替换
    return image.float(), mel_tensor.float() ,label

def get_all_image_paths(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths

class SequentialFramesAndAudioDataset(Dataset):

    def __init__(self, fold_path):
        self.fold_path = fold_path
        # self.frame_nums = frame_nums
        # self.frame_rate = frame_rate
        self.image_file_paths = get_all_image_paths(self.fold_path)

    def __len__(self):
        return len(self.image_file_paths)

    def __getitem__(self, idx):
        image_file = self.image_file_paths[idx]
        image, mel ,label = get_sequential_frames_and_audio(image_file)
        return image, mel, label


if __name__ == '__main__':
    train_dataset = SequentialFramesAndAudioDataset(
        fold_path='./_data_/cds_imgs',
        )
    for _, i in enumerate(train_dataset):
        image, mel, label = i
        print(image.shape)
        print(mel.shape)
        print(label)
        sleep(2)
