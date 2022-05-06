import os
import numpy as np
import matplotlib.pyplot as plt
#%%
x = np.load(r"D:\Faii\Dataset_flood\dataset-input-output-for training\X.npy")
y = np.load(r"D:\Faii\Dataset_flood\dataset-input-output-for training\y.npy")
#%%
n_samples = 9
while True:
    plt.close("all")
    index = np.random.permutation(len(x))[:9]
    img = x[index]
    label = y[index]
    fig, ax = plt.subplots(n_samples, 3, figsize=(2*4, 2*n_samples))
    for i in range(n_samples):
        ax[i, 0].imshow(img[i, :, :, 0], cmap="gray", vmin=0, vmax=1)
        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])
        ax[i, 1].imshow(img[i, :, :, 1], cmap="gray", vmin=0, vmax=1)
        ax[i, 1].set_xticks([])
        ax[i, 1].set_yticks([])
        ax[i, 2].imshow(img[i, :, :, 2], cmap="gray", vmin=0, vmax=1)
        ax[i, 2].set_xticks([])
        ax[i, 2].set_yticks([])
        
        # Show label
        ax[i, 0].yaxis.set_label_coords(-.5, 0)
        ax[i, 0].set_ylabel(f"Label: {label[i]}", rotation=0)

    plt.waitforbuttonpress()