import shutil
from glob import glob
import numpy as np
import os


def split_train_set():
    images_paths = glob(f'data/train/**/**.png')
    for im_path in images_paths:
        class_num = im_path.split('/')[2]
        im_name = im_path.split('/')[-1]
        os.makedirs(f'random_split_data/train/{class_num}', exist_ok=True)
        os.makedirs(f'random_split_data/val/{class_num}', exist_ok=True)
        if np.random.uniform() <= 0.8:
            shutil.copy(im_path, f'random_split_data/train/{class_num}/{im_name}')
        else:
            shutil.copy(im_path, f'random_split_data/val/{class_num}/{im_name}')


def copy_val():
    images_paths = glob(f'data/val/**/**.png')
    for im_path in images_paths:
        class_num = im_path.split('/')[2]
        im_name = im_path.split('/')[-1]
        os.makedirs(f'random_split_data/val/{class_num}', exist_ok=True)
        shutil.copy(im_path, f'random_split_data/val/{class_num}/{im_name}')


def main():
    # split_train_set()
    copy_val()


if __name__ == '__main__':
    main()
