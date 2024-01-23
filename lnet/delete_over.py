import os
from time import sleep
import numpy as np
from tqdm import tqdm

def delete_extra_frames(npy_file):
    directory=os.path.dirname(npy_file)
    # 加载.npy文件
    data = np.load(npy_file)

    # 获取.npy文件的长度
    length = len(data)
    print(length)

    # 获取目录中所有的.jpg文件
    image_files = [f for f in os.listdir(directory) if f.endswith('.jpg')]

    for image_file in image_files:
        image_name=int(image_file.split('.')[0])
        if image_name>=length:
            os.remove(os.path.join(directory, image_file))
            print('delete {}'.format(os.path.join(directory, image_file)))


def get_npy_files(directory):
    npy_files = []

    # 使用os.walk()遍历目录及其所有子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 如果文件是.npy文件，将其路径添加到列表中
            if file.endswith('.npy'):
                npy_files.append(os.path.join(root, file))

    return npy_files

if __name__ == '__main__':
    data_root = '/datadisk/huangkf/retalk_retrain/lnet/_data_/cds_preproc'
    npy_files=get_npy_files(data_root)
    for npy_file in tqdm(npy_files):
        delete_extra_frames(npy_file)
     