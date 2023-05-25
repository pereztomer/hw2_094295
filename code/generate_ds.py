import os
import random
import cv2
import numpy as np
import albumentations as A
from PIL import Image
from glob import glob


# gets PIL image and returns augmented PIL image
def augment_img(img):
    # only augment 3/4th the images
    if random.randint(1, 4) > 3:
        img = Image.fromarray(img)
        return img

    # morphological alterations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if random.randint(1, 5) == 1:
        # dilation because the image is not inverted
        img = cv2.erode(img, kernel, iterations=random.randint(1, 2))
    if random.randint(1, 6) == 1:
        # erosion because the image is not inverted
        img = cv2.dilate(img, kernel, iterations=random.randint(1, 1))

    transform = A.Compose([

        A.OneOf([
            # add black pixels noise
            A.OneOf([
                A.RandomRain(brightness_coefficient=1.0, drop_length=2, drop_width=2, drop_color=(0, 0, 0),
                             blur_value=1, rain_type='drizzle', p=0.05),
                A.RandomShadow(p=1),
                A.PixelDropout(p=1),
            ], p=0.9),

            # add white pixels noise
            A.OneOf([
                A.PixelDropout(dropout_prob=0.5, drop_value=255, p=1),
                A.RandomRain(brightness_coefficient=1.0, drop_length=2, drop_width=2, drop_color=(255, 255, 255),
                             blur_value=1, rain_type=None, p=1),
            ], p=0.9),
        ], p=1),

        # transformations
        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0, scale_limit=0.25, rotate_limit=2, border_mode=cv2.BORDER_CONSTANT,
                               value=(255, 255, 255), p=1),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=8, border_mode=cv2.BORDER_CONSTANT,
                               value=(255, 255, 255), p=1),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.15, rotate_limit=11, border_mode=cv2.BORDER_CONSTANT,
                               value=(255, 255, 255), p=1),
            A.Affine(shear=random.randint(-5, 5), mode=cv2.BORDER_CONSTANT, cval=(255, 255, 255), p=1)
        ], p=0.5),
        A.Blur(blur_limit=5, p=0.25),
    ])
    img = transform(image=img)['image']
    image = Image.fromarray(img)
    return image


def generate_ds(ds_path):
    im_paths = glob(f'{ds_path}/**/**.png')
    for im_p in im_paths:
        im = Image.open(im_p)
        aug_im = augment_img(im)
        im_folder = im_p.replace(im_p.split('/')[-1], '')
        im_name = im_p.split('/')[-1]
        aug_im.save(f"{im_folder}/aug_{im_name}")


def aug_ds(original_ds_path, new_ds_path):
    images_paths = glob(f'{original_ds_path}/**/**/**.png')
    for im_path in images_paths:
        im_name = im_path.split('/')[-1].replace('.png', '')
        im_class = im_path.split('/')[-2]
        im_set_type = im_path.split('/')[-3]
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if im_set_type == 'train':
            for i in range(5):
                aug_im = augment_img(im)
                os.makedirs(f'{new_ds_path}/{im_set_type}/{im_class}', exist_ok=True)
                aug_im.save(f'{new_ds_path}/{im_set_type}/{im_class}/{im_name}_aug_{i}.png')
        else:
            os.makedirs(f'{new_ds_path}/{im_set_type}/{im_class}', exist_ok=True)
            image = Image.fromarray(im)
            image.save(f'{new_ds_path}/{im_set_type}/{im_class}/{im_name}.png')


def main():
    aug_ds(original_ds_path='/home/user/PycharmProjects/hw2_094295/random_split_data',
           new_ds_path='/home/user/PycharmProjects/hw2_094295/aug_1_ds')
    # im_path = '/home/user/PycharmProjects/hw2_094295/random_split_data/train/i/ab9fb784-ce5d-11eb-b317-38f9d35ea60f.png'
    # im = Image.open(im_path)
    # im.show()
    # aug_im = augment_img(im)
    # aug_im.show()


if __name__ == '__main__':
    main()
