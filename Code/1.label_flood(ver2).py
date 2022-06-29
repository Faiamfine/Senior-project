import os
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

# open path
base_dir = "D:\Faii\Dataset_flood"
folder_for_visualize = os.path.join(base_dir, "crop_im_for_visualize")
# print(folder_dir)
folder_im = os.path.join(base_dir, "crop_im")

# pathsave
folder_save = os.path.join(base_dir, "label_picture")
os.makedirs(folder_save, exist_ok=True)

# Get names of all labeled images
list_labeled_images = [
    "_".join(file.split("_")[1:])[:-4] for file in os.listdir(folder_save)
]

print(list_labeled_images)

list_for_visualize = os.listdir(folder_for_visualize)
random.shuffle(list_for_visualize)

# extract all path
for i, img_name in enumerate(list_for_visualize):
    if img_name in list_labeled_images:
        print(f"Skip: {img_name}")
        continue

    img_path = os.path.join(folder_for_visualize, img_name)
    img_label = os.path.join(folder_im, img_name + ".npy")
    train_im = np.load(img_label)
    image1 = cv2.imread(img_path)
    image2 = cv2.imread(img_label)
    # print(img_path)
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image1)
        key = cv2.waitKey(0) & 0xFF
        # if the 'c' key is pressed, break from the loop
        if key == ord("c"):
            break
        elif key == ord("1"):
            np.save(os.path.join(folder_save, "flood_" + img_name + ".npy"), train_im)
            break
        elif key == ord("2"):
            np.save(
                os.path.join(folder_save, "nonflood_" + img_name + ".npy"), train_im
            )
            break
        elif key == ord("3"):
            np.save(os.path.join(folder_save, "notsure_" + img_name + ".npy"), train_im)
            break
        elif key == ord("q"):
            break
