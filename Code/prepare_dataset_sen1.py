# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 15:28:08 2022

@author: Teerasit_com4
"""
import datetime
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal

#%%
base_dir = r"D:\Faii\Dataset_flood\flood"
dirs = os.listdir(base_dir)
dirs = sorted(dirs, key=lambda x: float(x.split("_")[1]))
# arr_image #arr_dateim
os.makedirs(r"D:\Faii\cleaned_sorted_flood", exist_ok=True)
for file in dirs:
    if file.endswith(".xml"):
        continue
    print(file)
    path = os.path.join(base_dir, file)
    print(path)
    read_im = gdal.Open(path)
    arr_im = read_im.ReadAsArray()
    # extract date name file
    list_band_date = list()
    list_img = list()

    # count nan
    total_pixel = arr_im.shape[1] * arr_im.shape[2]
    for i in range(len(arr_im)):

        nan_count = np.isnan(arr_im[i]).sum()
        ratio_nan = nan_count / total_pixel
        if ratio_nan >= 0.2:
            continue
        # date time name file
        namefile = read_im.GetRasterBand(i + 1).GetDescription()
        if namefile.endswith("VV"):
            continue
        date = namefile.split("_")[4][0:8]
        date_time = datetime.datetime.strptime(date, "%Y%m%d")
        list_band_date.append(date_time)
        list_img.append(arr_im[i])
    list_band_date, list_img = zip(*sorted(zip(list_band_date, list_img)))
    list_img = np.array(list_img)
    # save raster
    if len(list_img.shape) == 2:
        list_img = np.expand_dims(list_img, 0)

    nodata = 0
    band = list_img.shape[0]
    row = list_img.shape[1]
    col = list_img.shape[2]
    driver = gdal.GetDriverByName("GTiff")
    output = driver.Create(
        rf"D:\Faii\cleaned_sorted_flood\{file}", col, row, band, gdal.GDT_Float32
    )
    output.SetProjection(read_im.GetProjection())
    output.SetGeoTransform(read_im.GetGeoTransform())
    for i in range(band):
        if list_band_date is not None:
            band_name = list_band_date[i]
            band_name = str(band_name.date())
            output.GetRasterBand(i + 1).SetDescription(band_name)
        output.GetRasterBand(i + 1).WriteArray(list_img[i, :, :])

        if nodata is not None:
            output.GetRasterBand(i + 1).SetNoDataValue(nodata)
        output.FlushCache()
    del output
    del driver
