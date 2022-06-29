# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 11:06:53 2022

@author: Teerasit_com4
"""
import io
import os
import pickle

import matplotlib.pyplot as plt

dict_result_ml = dict()
dict_result_al = dict()
#%%ml path extraction
root = r"D:\Faii\result\final"

for root, dirs, files in os.walk(base_ml):
    # print(root)
    for d in dirs:
        print(d)
        if "ml_training" in root:
            # print(root)
            for f in files:
                dir_name = os.path.join(root, f)
                print(dir_name)
                filename = f.split("_")[3]
                # print(filename)
                if filename == "acc":
                    with io.open(dir_name, "rb") as outputfile:
                        file_ml_acc = pickle.load(outputfile)
                elif filename == "loss":
                    with io.open(dir_name, "rb") as outputfile:
                        file_ml_loss = pickle.load(outputfile)
        # al path extracted
        elif "al_training" in dirs:
            for f in files:
                print(f)
                dir_name_al = os.path.join(root, f)
                # print(dir_name_al)
                filename_al = f[:].split("_")[0]
                # print(filename_al)
                if filename_al == "en":
                    specific_filename = f[:].split("_")[3]
                    if specific_filename == "val_acc":
                        with io.open(dir_name_al, "rb") as outputfile:
                            file_en_acc = pickle.load(outputfile)
                    elif specific_filename == "val_loss":
                        with io.open(dir_name_al, "rb") as outputfile:
                            file_en_loss = pickle.load(outputfile)
                elif filename_al == "lc":
                    if specific_filename == "val_acc":
                        with io.open(dir_name_al, "rb") as outputfile:
                            file_lc_acc = pickle.load(outputfile)
                    elif specific_filename == "val_loss":
                        with io.open(dir_name_al, "rb") as outputfile:
                            file_lc_loss = pickle.load(outputfile)
