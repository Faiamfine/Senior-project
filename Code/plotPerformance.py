# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 09:59:42 2022

@author: Teerasit_com4
"""


import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def readFile(main_dir, prefix, num):
    file_prefix = os.path.join(main_dir, prefix)
    output = []
    for k in range(num):
        file_name = f"{file_prefix}{k+1}"
        print(f"Reading: ", file_name)
        df = np.array(pd.read_pickle(file_name))
        output.append(df)
    output = np.array(output)
    stds = output.std(0)
    means = output.mean(0)
    return means, stds


if __name__ == "__main__":
    plt.close("all")
    entropy_dir = r"D:\Faii\result\AjTeerasit\Entropy1"
    ml_dir = r"D:\Faii\result\AjTeerasit\ml_training"
    u_dir = r"D:\Faii\result\AjTeerasit\U"
    values_lists = ["Train_Acc", "Train_Loss", "Val_Acc", "Val_Loss"]
    prefix_list = [
        "result_train_acc",
        "result_train_loss",
        "result_val_acc",
        "result_val_loss",
    ]
    prefix_names = {entropy_dir: "en_", ml_dir: "ml_", u_dir: "lc_"}

    for val, val_prefix in zip(values_lists, prefix_list):
        plt.figure()  # figsize=(15,10)
        for folder in [entropy_dir, u_dir, ml_dir]:
            main_folder = os.path.join(folder, val)
            file_prefix = prefix_names[folder] + val_prefix
            means, stds = readFile(main_folder, file_prefix, num=20)
            plt.errorbar(
                x=np.arange(len(means)) * 50 + 100,
                y=means,
                yerr=stds,
                label=prefix_names[folder][:2],
                markersize=8,
                capsize=30,
                fmt="o",
            )
        plt.title(val)
        if "acc" in val_prefix:
            plt.ylim(0.5, 1)
            plt.ylabel("Accuracy")
        else:
            plt.ylabel("Loss")

        plt.legend()
        plt.xticks()
        plt.xlabel("Number Of Samples")
        plt.grid()
    plt.show()
