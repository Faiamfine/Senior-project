import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

# to collect label
flood = list()
nonflood = list()
# filename
file_flood = list()
file_nonflood = list()


def get_roi(img):
    global image
    image = img.copy()
    clone = image.copy()
    window_width = int(img.shape[1] * 0.5)
    window_height = int(img.shape[0] * 0.5)

    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow("image", window_width, window_height)
    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(0) & 0xFF
        # if the 'c' key is pressed, break from the loop
        if key == ord("c"):
            break
        elif key == ord("1"):
            label = "flood"
            break
        elif key == ord("0"):
            label = "non-flooded"
            break
        elif key == ord("q"):
            label = "exit"
            break
    return label

    # label = None
    # try:


def plt_imshow(img):
    plt.imshow(img)
    plt.show()


# plt_imshow(image1)
count_n = np.random.randint(100000, size=1)[0]
count_y = np.random.randint(100000, size=1)[0]
base_dir = "D:\Faii"  # path for label and visualize
folder_dir = os.path.join(base_dir, "crop_im_for_visualize")
# print(folder_dir)

for img_name in os.listdir(folder_dir):
    img_path = os.path.join(folder_dir, img_name)
    image1 = cv2.imread(img_path)
    print(img_path)

    while True:
        label = get_roi(image1)
        if "flood" in label.lower():
            count_n += 1
            dir = r"D:\Faii\label-flood"
            filepath = os.path.join(dir, str(count_n))
            print(filepath)
            flood.append(label)
            file_flood.append(filepath)
        elif "nonflood" in label.lower():
            count_y += 1
            dir = r"D:\Faii\label-nonflood"
            filepath = os.path.join(dir, str(count_y))
            print(filepath)
            nonflood.append(label)
            file_nonflood.append(filepath)
        else:
            break
        print()
