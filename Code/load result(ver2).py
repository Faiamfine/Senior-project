# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 18:23:01 2022

@author: Teerasit_com4
"""
import io
import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
#%%
dict_result_ml = dict()
dict_result_al = dict()
#%%ml path extraction
root_ml = r'D:\Faii\result\final\ml_training'
root_al = r'D:\Faii\result\final\al_training'
# folder_ml = os.path.join(root,'ml_training')
# print(folder_ml)
# print(folder_al)
list_ml = os.listdir(root_ml)
# print(list_ml)  
for imgname_ml in tqdm(list_ml):
    img_ml_dir = os.path.join(root_ml,imgname_ml)  
    # print(img_ml_dir)
    assert os.path.isfile(img_ml_dir)
    filename = imgname_ml.split("\\")[-1]
    if filename == "val_acc":
        with io.open(img_ml_dir,'rb') as outputfile:
            file_ml_acc = pickle.load(outputfile)