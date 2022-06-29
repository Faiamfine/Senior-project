# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 22:19:33 2022

@author: Teerasit_com4
"""
import os

# crop 3 bands
import cv2
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal

x = 0  # cols
y = 0  # rows
window_size = 200
list_crop_im = list()
os.makedirs(r"D:\Faii\Dataset_flood\crop_im", exist_ok=True)
os.makedirs(r"แอแปปปปป", exist_ok=True)

# path_vv
base_dir_VV = r"D:\Faii\Dataset_flood\Satellite_image\1.1 cleaned_sorted_flood_VV"
base_dir_VH = r"D:\Faii\Dataset_flood\Satellite_image\1.2 cleaned_sorted_flood_VH"

#%%
dirs = os.listdir(base_dir_VV)

for file in dirs:
    print(file)
    path_VV = os.path.join(base_dir_VV, file)
    path_VH = os.path.join(base_dir_VH, file)
    print(path_VV)
    print(path_VH)
    img_VV = gdal.Open(path_VV)
    img_VH = gdal.Open(path_VH)
    num_band_VV = img_VV.RasterCount
    num_band_VH = img_VH.RasterCount
    # print(num_band)
    arr_im_VV = img_VV.ReadAsArray()
    arr_im_VH = img_VH.ReadAsArray()

    arr_im_VV = (arr_im_VV - np.nanmin(arr_im_VV)) * (1 - 0) / (
        np.nanmax(arr_im_VV) - np.nanmin(arr_im_VV)
    ) + 0
    arr_im_VH = (arr_im_VH - np.nanmin(arr_im_VH)) * (1 - 0) / (
        np.nanmax(arr_im_VH) - np.nanmin(arr_im_VH)
    ) + 0
    rows = arr_im_VV.shape[1]
    cols = arr_im_VV.shape[2]
    step_size_row = rows // window_size
    step_size_col = cols // window_size
    # arr_im.shape[0]=number of band im
    for i in range(arr_im_VV.shape[0] - 2):
        for x in range(0, step_size_col):
            for y in range(0, step_size_row):
                crop_im_VV = arr_im_VV[
                    i : i + 3,
                    y * window_size : (y * window_size) + window_size,
                    x * window_size : (x * window_size) + window_size,
                ]
                crop_im_VH = arr_im_VH[
                    i : i + 3,
                    y * window_size : (y * window_size) + window_size,
                    x * window_size : (x * window_size) + window_size,
                ]

                crop_im_VV = np.moveaxis(crop_im_VV, 0, -1)
                crop_im_VH = np.moveaxis(crop_im_VH, 0, -1)
                crop_im_VVVH = crop_im_VV - crop_im_VH
                crop_im_VVVH = (crop_im_VVVH - crop_im_VVVH.min()) / (
                    crop_im_VVVH.max() - crop_im_VVVH.min()
                )
                # list_crop_im.append(crop_im)
                # stack
                crop_im = np.concatenate([crop_im_VV, crop_im_VH], axis=-1)
                np.save(
                    os.path.join(
                        r"D:\Faii\Dataset_flood\crop_im",
                        f'{file.split(".")[0]}_{y*window_size}-{(y*window_size)+window_size}_{x*window_size}-{(x*window_size)+window_size}_{i}-{i+3}.png',
                    ),
                    crop_im,
                )
                # plt
                plt.close("all")
                fig, ax = plt.subplots(ncols=3)

                ax[0].imshow(crop_im_VV[:, :, 0], cmap="gray", vmin=0, vmax=1)
                ax[1].imshow(crop_im_VV[:, :, 1], cmap="gray", vmin=0, vmax=1)
                ax[2].imshow(crop_im_VV[:, :, 2], cmap="gray", vmin=0, vmax=1)
                fig.savefig(
                    os.path.join(
                        r"D:\Faii\Dataset_flood\crop_im_for_visualize",
                        f'{file.split(".")[0]}_{y*window_size}-{(y*window_size)+window_size}_{x*window_size}-{(x*window_size)+window_size}_{i}-{i+3}.png',
                    ),
                    bbox_inches="tight",
                )
                print(
                    (i, i + 3),
                    y * window_size,
                    (y * window_size) + window_size,
                    x * window_size,
                    (x * window_size) + window_size,
                )

    # crop_im = arr_im[:,y:y+h,x:x+w]
    # cv2.imshow('Image', crop_im)
# #%%
# plt.close('all')
# plt.figure()
# plt.imshow(arr_im[i])
# for i in range(len(list_crop_im[:9])):
#     image = list_crop_im[i]
#     image = (image-image.min())*(1-0)/(image.max()-image.min())+0

#     plt.figure()
#     plt.imshow(np.moveaxis(image,0,-1))
