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


#%%
def make_model(input_shape, num_classes):
    # ResNet Model
    base_model = ResNet50V2(include_top=False, weights=None, input_shape=input_shape)

    # Connect layers
    inputs = keras.Input(shape=(input_shape[0], input_shape[1], 3))
    x = layers.experimental.preprocessing.RandomFlip("horizontal")(inputs)
    x = layers.experimental.preprocessing.RandomFlip("vertical")(x)
    x = base_model(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    # A Dense classifier with a single unit (binary classification)
    x = layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    return model


def get_callbacks():
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(
            r"D:\Faii\Dataset_flood\model", "activelearningv2-LC-resnet50v2.h5"
        ),
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


#%%
# list_x = list()
# list_y = list()
# list_name = list()
# root = 'D:\Faii\Dataset_flood'
# folder_img = os.path.join(root,'label_picture')
# # print(folder_dir)
# list_img_name = os.listdir(folder_img)
# random.shuffle(list_img_name)
# # #%%
# for img_name in tqdm(list_img_name):
#     img_dir = os.path.join(folder_img, img_name)
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
#     list_x.append(X)
#     list_y.append(y)
#     list_name.append(img_name)
# list_X = np.array(list_x)
# list_y = np.array(list_y)
# list_name = np.array(list_name)
# np.save(r"D:\Faii\Dataset_flood\dataset-input-output-for training\X.npy", list_x)
# np.save(r"D:\Faii\Dataset_flood\dataset-input-output-for training\y.npy", list_y)
# np.save(r"D:\Faii\Dataset_flood\dataset-input-output-for training\img_name.npy", list_name)
#%%
x_pool = np.load(r"D:\Faii\Dataset_flood\dataset-input-output-for training\X.npy")[
    :, :, :, :3
]
y_pool = np.load(r"D:\Faii\Dataset_flood\dataset-input-output-for training\y.npy")
name_pool = np.load(
    r"D:\Faii\Dataset_flood\dataset-input-output-for training\img_name.npy"
)

#%%
scaler = StandardScaler()
x_pool = scaler.fit_transform(x_pool.reshape(-1, x_pool.shape[-1])).reshape(
    x_pool.shape
)
#%% Define number of train data
num_sample = len(x_pool)
num_train = 100
num_val = 400
num_pool = num_sample - num_train - num_val

# Validation samples
x_val = x_pool[0:num_val]
y_val = y_pool[0:num_val]

# Training samples
x_train = x_pool[num_val : num_val + num_train]
y_train = y_pool[num_val : num_val + num_train]

x_pool = x_pool[num_val + num_train :]
y_pool = y_pool[num_val + num_train :]
name_pool = name_pool[num_val + num_train :]

# change to np.array
x_train = np.array(x_train)
y_train = np.array(y_train)
x_pool = np.array(x_pool)
y_pool = np.array(y_pool)
name_pool = np.array(name_pool)
x_val = np.array(x_val)
y_val = np.array(y_val)
#%% Complie and save initial model
model = make_model(input_shape=(200, 200, 3), num_classes=2)
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.save_weights(
    os.path.join(
        r"D:\Faii\Dataset_flood\model", "activelearning-LC-resnet50v2(initial).h5"
    )
)
#%% Define active learning parameters
num_add = 50
num_iter = 15
num_epochs = 100
#%% Start active learning
acc = []
val_acc = []
loss = []
val_loss = []
#%%
for k in range(num_iter):
    print(f"Iteration [{k}] Train Sample Size: {len(x_train)}.")
    # Reset weights
    model.load_weights(
        os.path.join(
            r"D:\Faii\Dataset_flood\model", "activelearning-LC-resnet50v2(initial).h5"
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
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
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
    predict_prob = model.predict(x_pool, batch_size=16).flatten()
    u = np.abs(predict_prob - 0.5)
    print(u.min())

    # Add training samples based on "U"
    sorted_indices = np.argsort(u)
    adding_index = sorted_indices[0:num_add]

    # Show least confiences
    img_u = scaler.inverse_transform(
        x_pool[adding_index].reshape(-1, x_pool[adding_index].shape[-1])
    ).reshape(x_pool[adding_index].shape)
    label_u = y_pool[adding_index]
    name_u = name_pool[adding_index]
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
            f"Label: {label_u[i]}\nU:{u[sorted_indices][i]:.2f}\n{name_u[i]}",
            rotation=0,
        )
    fig.savefig(
        os.path.join(r"D:\Faii\result_each_iter\LC", f"iter_{k}.png"),
        bbox_inches="tight",
    )

    # Add data
    x_train = np.concatenate((x_train, x_pool[adding_index]), axis=0)
    y_train = np.concatenate((y_train, y_pool[adding_index]), axis=0)
    x_pool = np.delete(x_pool, adding_index, axis=0)
    y_pool = np.delete(y_pool, adding_index, axis=0)
    name_pool = np.delete(name_pool, adding_index, axis=0)
#%%
# summarize history for accuracy
plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.grid()
plt.savefig(
    os.path.join(
        r"D:\Faii\result\Result_final_al", "acc-activelearning-LC-resnet50v2.png"
    ),
    bbox_inches="tight",
)

# summarize history for loss
plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.grid()
plt.savefig(
    os.path.join(
        r"D:\Faii\result\Result_final_al", "loss-activelearning-LC-resnet50v2.png"
    ),
    bbox_inches="tight",
)
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
plt.savefig(
    os.path.join(r"D:\Faii\result\Result_final_al", "each_iter.png"),
    bbox_inches="tight",
)
#%%
import pickle

# loss
with open(r"result_val_loss", "wb") as f:
    pickle.dump(list_val_loss, f)
    pickle.dump(list_val_acc, f)
with open(r"result_val_loss", "rb") as f:
    val_acc = pickle.load(f)
print(val_acc)
