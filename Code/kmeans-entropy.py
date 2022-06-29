import io
import os
import pickle
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import tensorflow as tf
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


# %%
def make_model(input_shape):
    # ResNet Model
    base_model = ResNet50V2(include_top=False, weights=None, input_shape=input_shape)

    # Connect layers
    inputs = keras.Input(shape=(input_shape[0], input_shape[1], 3))
    x = layers.experimental.preprocessing.RandomFlip("horizontal")(inputs)
    x = layers.experimental.preprocessing.RandomFlip("vertical")(x)
    x_feature = base_model(x)
    x_out = keras.layers.GlobalAveragePooling2D()(x_feature)
    # A Dense classifier with a single unit (binary classification)
    x_out1 = layers.Dropout(0.5)(x_out)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x_out1)
    model = keras.Model(inputs, outputs)
    model_feat = keras.Model(inputs, x_out)
    return model, model_feat


def get_callbacks():
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(
            r"D:\Faii\Dataset_flood\model", "activelearning-entropy-resnet50v2.h5"
        ),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=1
    )
    early = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )
    return model_checkpoint, reduce_lr, early


def kMeanSamples(x_unlabeled, model_feat, model_class, num_samples=50, y_lab=None):
    prob = model_class.predict(x_unlabeled, batch_size=32).flatten()
    x_feat = model_feat.predict(x_unlabeled, batch_size=32).reshape(len(prob), -1)
    kmeans = KMeans(n_clusters=num_samples, random_state=0)
    # s = 1 - np.abs(0.5 -  prob)*2
    entropy1 = (-prob) * np.log(prob)
    entropy0 = (prob - 1) * np.log(1 - prob)
    s = entropy1 + entropy0
    s[prob <= 1e-6] = 0
    s[prob >= 1 - 1e-6] = 0
    # entropy = entropy1 + entropy0
    # entropy1 = (-prob)*np.log2(pprob)
    # entropy0 = (prob-1)*np.log2(1-prob)
    # entropy = entropy1 + entropy0
    # entropy[predict_prob <= 1e-6] = 0
    # entropy[predict_prob >= 1-1e-6] = 0
    labels = kmeans.fit_predict(x_feat, sample_weight=s)
    clas_lab = (prob > 0.5).astype("int")
    print(f"There are {clas_lab.mean()*100}% with predicted flood.")
    centers = kmeans.cluster_centers_
    sample_ids = []
    for lb in np.unique(labels):
        id_labels = np.where(labels == lb)[0]
        sample_lb = x_feat[id_labels, :]
        center_lb = centers[lb, :]
        dis = ((sample_lb - center_lb.reshape(1, -1)) ** 2).sum(1)
        id_min = np.where(dis == dis.min())[0][0]
        print(
            f"We select {id_labels[id_min]} from {len(id_labels)} for Cluster {lb} with s={s[id_labels[id_min]]}."
        )
        print(
            f"Image is labeled with {y_lab[id_labels[id_min]]} and prob: {prob[id_labels[id_min]]}"
        )
        sample_ids.append(id_labels[id_min])
    return np.array(sample_ids), x_feat, centers


# %%
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
# %%
for i in range(1, 10):
    print(f"round[{i}]")
    x_pool = np.load(r"D:\Faii\Dataset_flood\dataset-input-output-for training\X.npy")[
        :, :, :, :3
    ]
    y_pool = np.load(r"D:\Faii\Dataset_flood\dataset-input-output-for training\y.npy")
    name_pool = np.load(
        r"D:\Faii\Dataset_flood\dataset-input-output-for training\img_name.npy"
    )
    shuffle_index = np.random.permutation(len(x_pool))

    x_pool = x_pool[shuffle_index]
    y_pool = y_pool[shuffle_index]
    # %%
    scaler = StandardScaler()
    x_pool = scaler.fit_transform(x_pool.reshape(-1, x_pool.shape[-1])).reshape(
        x_pool.shape
    )

    num_train = 100
    num_validate = 400
    x_train = x_pool[num_validate : num_validate + num_train]
    y_train = y_pool[num_validate : num_validate + num_train]
    x_val = x_pool[0:num_validate]
    y_val = y_pool[0:num_validate]
    # delete validate samples out
    x_pool = x_pool[num_validate + num_train :]
    y_pool = y_pool[num_validate + num_train :]
    # %% Complie and save initial model
    model, model_feat = make_model(input_shape=(200, 200, 3))
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.save_weights(
        os.path.join(
            r"D:\Faii\Dataset_flood\model",
            "activelearning-entropy-resnet50v2(initial).h5",
        )
    )
    model_feat.save_weights(
        os.path.join(
            r"D:\Faii\Dataset_flood\model",
            "activelearning-entropy-resnet50v2(initial)_feat.h5",
        )
    )
    # %% Define active learning parameters
    num_add = 50
    num_iter = 15
    num_epochs = 100
    # %% Start active learning
    acc = []
    val_acc = []
    loss = []
    val_loss = []
    # %%
    for k in range(num_iter):
        print(f"Iteration [{k}] Train Sample Size: {len(x_train)}.")
        # Reset weights
        # model.load_weights(os.path.join(
        #    r'D:\Faii\Dataset_flood\model', "activelearning-entropy-resnet50v2(initial).h5"))
        # model_feat.load_weights(os.path.join(
        #    r'D:\Faii\Dataset_flood\model', "activelearning-entropy-resnet50v2(initial)_feat.h5"))

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

        adding_index, features, centers = kMeanSamples(
            x_pool, model_feat, model, num_samples=50, y_lab=y_pool
        )

        print(f"there are {y_pool[adding_index].mean()*100:0.2f} % of flooded images.")
        # Add data
        x_train = np.concatenate((x_train, x_pool[adding_index]), axis=0)
        y_train = np.concatenate((y_train, y_pool[adding_index]), axis=0)
        x_pool = np.delete(x_pool, adding_index, axis=0)
        y_pool = np.delete(y_pool, adding_index, axis=0)
    # %%
    list_val_acc = []
    list_val_loss = []
    list_train_acc = []
    list_train_loss = []

    for num_iter in range(len(acc)):
        print(num_iter, val_loss[num_iter][-1], val_acc[num_iter][-1])
        list_val_acc.append(val_acc[num_iter][-1])
        list_val_loss.append(val_loss[num_iter][-1])
        list_train_acc.append(acc[num_iter][-1])
        list_train_loss.append(loss[num_iter][-1])
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(list_val_loss, label="Validation Loss")
    plt.plot(list_train_loss, label="Train Loss")
    plt.legend()
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(list_val_acc, label="Validation Accuracy")
    plt.plot(list_train_acc, label="Train Accuracy")
    plt.legend()
    plt.grid()
    plt.show()
    # plt.savefig('D:\Faii\result\Aj.Teerasit\entropy')

    with io.open("kmean_en_result_val_acc" + str(i), "wb") as f:
        pickle.dump(list_val_acc, f)

    with io.open("kmean_en_result_val_loss" + str(i), "wb") as f:
        pickle.dump(list_val_loss, f)

    with io.open("kmean_en_result_train_acc" + str(i), "wb") as f:
        pickle.dump(list_train_acc, f)

    with io.open("kmean_en_result_train_loss" + str(i), "wb") as f:
        pickle.dump(list_train_loss, f)

    # clearing session
    del model
    del x_train
    del y_train
    del list_val_acc
    del list_val_loss
    del list_train_acc
    del list_train_loss
    K.clear_session()
