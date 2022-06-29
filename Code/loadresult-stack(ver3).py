# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 21:23:23 2022

@author: Teerasit_com4
"""
import io
import json
import os
import pickle

from tqdm import tqdm


def read_pickle(path):
    with io.open(path, "rb") as f:
        data = pickle.load(f)
    return data


def read_data(folder, file):
    dict_data = {}
    for ml_name in file:
        print(ml_name.split("_")[-1])
        ml_ac_dir = os.path.join(folder, ml_name)
        ml_acc_val = read_pickle(ml_ac_dir)
        dict_data[f'trial_{ml_name.split("_")[-1]}'] = ml_acc_val
    return dict_data
    # for en_name in file:
    #     print(ml_name.split("_")[-1])


#%%
# ml
root_ml = r"D:\Faii\result\final\ml_training"
ml_acc = os.path.join(root_ml, "val_acc")
ml_loss = os.path.join(root_ml, "val_loss")
list_ml_acc = os.listdir(ml_acc)
list_ml_loss = os.listdir(ml_loss)
#%%
root_al = r"D:\Faii\result\final\al_training\entropy"
en_acc = os.path.join(root_al, "val_acc")
en_loss = os.path.join(root_al, "val_loss")
list_en_acc = os.listdir(en_acc)
list_en_loss = os.listdir(en_loss)

root_al_lc = r"D:\Faii\result\final\al_training\lc"
lc_acc = os.path.join(root_al_lc, "val_acc")
lc_loss = os.path.join(root_al_lc, "val_loss")
list_lc_acc = os.listdir(lc_acc)
list_lc_loss = os.listdir(lc_loss)
#%%
ml_acc_file = read_data(ml_acc, list_ml_acc)
json.dump(ml_acc_file, open(r"D:\Faii\result\final\ml_acc.json", "w"))
ml_loss_file = read_data(ml_loss, list_ml_loss)
json.dump(ml_loss_file, open(r"D:\Faii\result\final\ml_loss.json", "w"))
#%%
en_acc_file = read_data(en_acc, list_en_acc)
json.dump(en_acc_file, open(r"D:\Faii\result\final\en_acc.json", "w"))
en_loss_file = read_data(en_loss, list_en_loss)
json.dump(en_loss_file, open(r"D:\Faii\result\final\en_loss.json", "w"))
#%%
lc_acc_file = read_data(lc_acc, list_lc_acc)
json.dump(lc_acc_file, open(r"D:\Faii\result\final\lc_acc.json", "w"))
lc_loss_file = read_data(lc_loss, list_lc_loss)
json.dump(lc_loss_file, open(r"D:\Faii\result\final\lc_loss.json", "w"))
#%%
