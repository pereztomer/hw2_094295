import glob
import os
import cv2
import uuid
import json
import random

def rename_images_to_uuids(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            if os.path.isfile(file_path):
                file_extension = os.path.splitext(filename)[1]
                new_filename = str(uuid.uuid4()) + file_extension
                new_file_path = os.path.join(root, new_filename)
                os.rename(file_path, new_file_path)
                print(f"Renamed {filename} to {new_filename}")

    print("Image renaming completed.")


def delete_random_split(root_dir):
    images_paths = glob.glob(root_dir)
    sampled_list = random.sample(images_paths, 260)
    for item in sampled_list:
        os.remove(item)





def convert_images_to_grayscale(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)

            # Check if the file is an image (you can modify the list of supported image file extensions if needed)
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                try:
                    # Read the image
                    image = cv2.imread(file_path)

                    # Convert the image to grayscale
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    # Save the grayscale image with the same file name
                    cv2.imwrite(file_path, gray_image)

                    print(f"Converted {file_path} to grayscale.")
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")

    print("Conversion to grayscale completed.")


def num_list(src):
    srcs = glob.glob(src+'\\*')
    nums = [src.split('_')[-1][:-4] for src in srcs]
    with open("test_file.json", 'w') as json_file:
        json.dump(nums, json_file)

if __name__ == '__main__':
    # num_list("D:\\blue_gloves\\test_images")
    delete_random_split("C:\\Users\\RoiPapo\\PycharmProjects\\hw2_094295\\code\\aug_5_ds\\train\\**\\**.png")
    # rename_images_to_uuids('C:\\Users\\RoiPapo\\PycharmProjects\\hw2_094295\\code\\aug_3_ds\\train')

