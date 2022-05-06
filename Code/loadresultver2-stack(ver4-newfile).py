# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 09:40:56 2022

@author: Teerasit_com4
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 21:23:23 2022

@author: Teerasit_com4
"""
import os 
import io
import json
import pickle
from tqdm import tqdm

#"D:\Faii\result\AjTeerasit\Entropy\train_acc\en_result_train_acc1"
def read_pickle(path):
    with io.open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def read_data(folder,file):
    dict_data = {}
    for ml_name in file:
        print(ml_name.split("_")[-1])
        ml_ac_dir = os.path.join(folder,ml_name)
        ml_acc_val = read_pickle(ml_ac_dir)
        dict_data[f'trial_{ml_name.split("_")[-1]}'] = ml_acc_val
    return dict_data
#%%
# ml 
root_ml = r'D:\Faii\result\AjTeerasit\ml_training'
ml_acc = os.path.join(root_ml,'val_acc')
ml_train_acc = os.path.join(root_ml,'train_acc')
ml_loss = os.path.join(root_ml,'val_loss')
ml_train_loss = os.path.join(root_ml,'train_loss')
list_ml_acc = os.listdir(ml_acc)
list_ml_train_acc = os.listdir(ml_train_acc)
list_ml_loss = os.listdir(ml_loss)
list_ml_train_loss = os.listdir(ml_train_loss)
#%%
root_al = r'D:\Faii\result\AjTeerasit\Entropy1'
en_acc = os.path.join(root_al,'val_acc')
en_train_acc = os.path.join(root_al,'train_acc')
en_loss = os.path.join(root_al,'val_loss')
en_train_loss = os.path.join(root_al,'train_loss')
list_en_acc = os.listdir(en_acc)
list_en_train_acc = os.listdir(en_train_acc)
list_en_loss = os.listdir(en_loss)
list_en_train_loss = os.listdir(en_train_loss)

# root_al_lc =  r'D:\Faii\result\AjTeerasit\U'
# lc_acc = os.path.join(root_al_lc,'val_acc')
# lc_train_acc = os.path.join(root_al_lc,'train_acc')
# lc_loss = os.path.join(root_al_lc,'val_loss')
# lc_train_loss = os.path.join(root_al_lc,'train_loss')
# list_lc_acc = os.listdir(lc_acc)
# list_lc_train_acc = os.listdir(lc_train_acc)
# list_lc_loss = os.listdir(lc_loss)
# list_lc_train_loss = os.listdir(lc_train_loss)
#%%
# ml_acc_file = read_data(ml_acc,list_ml_acc)
# ml_train_acc_file = read_data(ml_train_acc,list_ml_train_acc)
# json.dump(ml_acc_file,open(r'D:\Faii\result\AjTeerasit\ml_acc.json','w'))
# json.dump(ml_train_acc_file,open(r'D:\Faii\result\AjTeerasit\ml_train_acc.json','w'))
# ml_loss_file = read_data(ml_loss,list_ml_loss)
# ml_train_loss_file = read_data(ml_train_loss,list_ml_train_loss)
# json.dump(ml_loss_file,open(r'D:\Faii\result\AjTeerasit\ml_loss.json','w'))
# json.dump(ml_train_loss_file,open(r'D:\Faii\result\AjTeerasit\ml_train_loss.json','w'))
#%%
en_acc_file = read_data(en_acc,list_en_acc)
en_train_acc_file = read_data(en_train_acc,list_en_train_acc)
json.dump(en_acc_file,open(r'D:\Faii\result\AjTeerasit\en_acc2.json','w'))
json.dump(en_acc_file,open(r'D:\Faii\result\AjTeerasit\en_train_acc2.json','w'))
en_loss_file = read_data(en_loss,list_en_loss)
en_train_loss_file = read_data(en_train_loss,list_en_train_loss)
json.dump(en_loss_file,open(r'D:\Faii\result\AjTeerasit\en_loss2.json','w'))
json.dump(en_train_loss_file,open(r'D:\Faii\result\AjTeerasit\en_train_loss2.json','w'))
#%%
# lc_acc_file = read_data(lc_acc,list_lc_acc)
# lc_train_acc_file = read_data(lc_train_acc,list_lc_train_acc)
# json.dump(lc_acc_file,open(r'D:\Faii\result\AjTeerasit\lc_acc.json','w'))
# json.dump(lc_train_acc_file,open(r'D:\Faii\result\AjTeerasit\lc_train_acc.json','w'))
# lc_loss_file = read_data(lc_loss,list_lc_loss)
# lc_train_loss_file = read_data(lc_train_loss,list_lc_train_loss)
# json.dump(lc_loss_file,open(r'D:\Faii\result\AjTeerasit\lc_loss.json','w'))
# json.dump(lc_train_loss_file,open(r'D:\Faii\result\AjTeerasit\lc_train_loss.json','w'))
#%%
