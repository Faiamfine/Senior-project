# # -*- coding: utf-8 -*-
# """
# Created on Tue Feb  8 21:36:18 2022

# @author: Teerasit_com4
# """
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# # ##%%
list_X = list()
list_y = list()
base_dir = "D:\Faii\Dataset_flood"
folder_dir = os.path.join(base_dir, "label_picture")
# print(folder_dir)
list_img_name = os.listdir(folder_dir)
random.shuffle(list_img_name)
#%%
# for img_name in tqdm(list_img_name):
#     img_dir = os.path.join(folder_dir, img_name)
#     filename = img_name.split("_")[0]
#     # print(img_dir)
#     X= np.load(img_dir)
#     if np.isnan(X).any():
#         continue
#     if filename == 'flood':
#         y = 1
#     elif filename == 'notsure':
#         continue
#     else:
#         y = 0
#     list_X.append(X)
#     list_y.append(y)
# list_X = np.array(list_X)
# list_y = np.array(list_y)
# np.save(r"D:\Faii\Dataset_flood\dataset-input-output-for training\X.npy",list_X)
# np.save(r"D:\Faii\Dataset_flood\dataset-input-output-for training\y.npy",list_y)
#%%
# pool_data_sample=np.load(r"D:\Faii\Dataset_flood\dataset-input-output-for training\X.npy")
pool_data_sample = np.load(
    r"D:\Faii\Dataset_flood\dataset-input-output-for training\X.npy"
)[:, :, :, :3]
pool_img_label = np.load(
    r"D:\Faii\Dataset_flood\dataset-input-output-for training\y.npy"
)
# pool_img_label = to_categorical(pool_img_label, 2, dtype="uint8")
#%%
scaler = StandardScaler()
pool_data_sample = scaler.fit_transform(
    pool_data_sample.reshape(-1, pool_data_sample.shape[-1])
).reshape(pool_data_sample.shape)
#%%
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    # x = layers.experimental.preprocessing.RandomFlip("horizontal")(inputs)

    # Entry block
    # x = layers.experimental.preprocessing.Rescaling(1.0/255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        # for size in [128]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    activation = "softmax"
    units = num_classes

    # x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


def get_callbacks():
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(r"D:\Faii\Dataset_flood\model", "activelearning-resnet50v2.h5"),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=6, min_lr=1e-8, verbose=1
    )
    early = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )
    return model_checkpoint, reduce_lr, early


# # Downsampling 25000 -> 15000
# pool_data_sample = pool_data_sample[:2394]
# pool_img_label = pool_img_label[:20000]
#%%
image_size = (200, 200)
from tensorflow.keras.layers import (Conv2D, Dense, Flatten,
                                     GlobalAveragePooling2D, MaxPooling2D)
#%% Define model
# model = make_model(input_shape=image_size+(6,), num_classes=2)
from tensorflow.keras.models import Sequential

# create model
model = Sequential()
# add model layers
model.add(Conv2D(16, kernel_size=3, activation="relu", input_shape=(200, 200, 3)))
model.add(Conv2D(32, kernel_size=3, activation="relu"))
model.add(Conv2D(32, kernel_size=3, strides=2, activation="relu"))

model.add(Conv2D(32, kernel_size=3, activation="relu"))
model.add(Conv2D(64, kernel_size=3, activation="relu"))
model.add(Conv2D(64, kernel_size=3, strides=2, activation="relu"))
model.add(Conv2D(128, kernel_size=3, strides=2, activation="relu"))
model.add(Conv2D(128, kernel_size=3, strides=2, activation="relu"))

model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()
# categorical_crossentropy
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#%% Define number of train data
pool_sample = len(pool_data_sample)
num_train = 100
num_validate = 400
num_pool = pool_sample - num_train - num_validate

# Validation samples
validate_data_sample = pool_data_sample[0:num_validate]
validate_img_label = pool_img_label[0:num_validate]

# Training samples
train_data_sample = pool_data_sample[num_validate : num_validate + num_train]
train_img_label = pool_img_label[num_validate : num_validate + num_train]

pool_data_sample = pool_data_sample[num_validate + num_train :]
pool_img_label = pool_img_label[num_validate + num_train :]

# change to np.array
train_data_sample = np.array(train_data_sample)
train_img_label = np.array(train_img_label)
pool_data_sample = np.array(pool_data_sample)
pool_img_label = np.array(pool_img_label)
validate_data_sample = np.array(validate_data_sample)
validate_img_label = np.array(validate_img_label)
#%% Complie and save initial model
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.save_weights(
    os.path.join(r"D:\Faii\Dataset_flood\model", "activelearning-custom(initial).h5")
)
#%% Define active learning parameters
num_samples = 50
num_iter = 15
num_epochs = 100
#%% Start active learning
acc = []
val_acc = []
loss = []
val_loss = []
#%%
for k in range(num_iter):
    print(f"Iteration [{k}] Train Sample Size: {len(train_data_sample)}.")
    # Reset weights
    model.load_weights(
        os.path.join(
            r"D:\Faii\Dataset_flood\model", "activelearning-custom(initial).h5"
        )
    )

    # Get callbacks
    model_checkpoint, reduce_lr, early = get_callbacks()
    callbacks = [model_checkpoint, reduce_lr, early]

    # Complie model
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics="accuracy",
    )

    # Train
    history = model.fit(
        x=train_data_sample,
        y=train_img_label,
        validation_data=(validate_data_sample, validate_img_label),
        batch_size=32,
        epochs=num_epochs,
        callbacks=callbacks,
    )
    # Store the history of each iteration
    acc.append(history.history["accuracy"])
    val_acc.append(history.history["val_accuracy"])
    loss.append(history.history["loss"])
    val_loss.append(history.history["val_loss"])

    # Calculate "U"
    predict_prob = model.predict(pool_data_sample, batch_size=16).flatten()
    u = np.abs(predict_prob - 0.5)
    print(u.min())

    # Add training samples based on "U"
    sorted_indices = np.argsort(u)
    adding_index = sorted_indices[0:num_samples]

    # Show least confiences
    img_u = scaler.inverse_transform(
        pool_data_sample[adding_index].reshape(
            -1, pool_data_sample[adding_index].shape[-1]
        )
    ).reshape(pool_data_sample[adding_index].shape)
    label_u = pool_img_label[adding_index]
    plt.close("all")
    n_samples = 5
    fig, ax = plt.subplots(n_samples, 3, figsize=(2 * 3 + 1, 2 * n_samples))
    for i in range(n_samples):
        ax[i, 0].imshow(img_u[i, :, :, 0], cmap="gray", vmin=0, vmax=1)
        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])
        ax[i, 1].imshow(img_u[i, :, :, 1], cmap="gray", vmin=0, vmax=1)
        ax[i, 1].set_xticks([])
        ax[i, 1].set_yticks([])
        ax[i, 2].imshow(img_u[i, :, :, 2], cmap="gray", vmin=0, vmax=1)
        ax[i, 2].set_xticks([])
        ax[i, 2].set_yticks([])

        # Show label
        ax[i, 0].yaxis.set_label_coords(-0.5, 0)
        ax[i, 0].set_ylabel(
            f"Label: {label_u[i]}\nU:{u[sorted_indices][i]:.2f}", rotation=0
        )
    fig.savefig(
        os.path.join(r"D:\Faii\result_each_iter", f"iter_{k}.png"), bbox_inches="tight"
    )

    # Add data
    train_data_sample = np.concatenate(
        (train_data_sample, pool_data_sample[adding_index]), axis=0
    )
    train_img_label = np.concatenate(
        (train_img_label, pool_img_label[adding_index]), axis=0
    )
    pool_data_sample = np.delete(pool_data_sample, adding_index, axis=0)
#%%
predict_prob = model.predict(pool_data_sample, batch_size=16).flatten()
u = np.abs(predict_prob - 0.5)
sorted_indices = np.argsort(u)
adding_index = sorted_indices[0:num_samples]
#%%

#%%

#%%
# for i in range(len(loss)):
#     if np.min(loss):
#         print(loss)
#%%
list_val_acc = []
list_val_loss = []

for num_iter in range(len(acc)):
    print(num_iter, min(val_loss[num_iter]), max(val_acc[num_iter]))
    list_val_acc.append(max(val_acc[num_iter]))
    list_val_loss.append(min(val_loss[num_iter]))


plt.figure()
plt.plot(list_val_loss, label="validation loss")
plt.plot(list_val_acc, label="validation accuracy")
plt.xlabel("Iteration")
plt.legend(loc="best")
#%%
# num_loss = len(loss)
# for iteration ,num_loss in enumerate(num_loss):
#     print(f'{iteration}->{loss}')
