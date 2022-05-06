# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 07:32:44 2022

@author: Teerasit_com4
"""
import os
import io
import cv2
import pickle
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import backend as K 
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50V2
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
        os.path.join(r'D:\Faii\Dataset_flood\model', "activelearningv2-LC-resnet50v2.h5"),
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=6, min_lr=1e-8, verbose=1
    )
    early = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
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
for i in range(11,21):
    print(f'round[{i}]')
    x_pool = np.load(r"D:\Faii\Dataset_flood\dataset-input-output-for training\X.npy")[:, :, :, :3]
    y_pool = np.load(r"D:\Faii\Dataset_flood\dataset-input-output-for training\y.npy")
    name_pool = np.load(r"D:\Faii\Dataset_flood\dataset-input-output-for training\img_name.npy")
    shuffle_index = np.random.permutation(len(x_pool))
    
    x_pool = x_pool[shuffle_index] 
    y_pool = y_pool[shuffle_index]  
    #%%
    scaler = StandardScaler()
    x_pool = scaler.fit_transform(x_pool.reshape(-1, x_pool.shape[-1])).reshape(x_pool.shape)
    #%% Define number of train data
    num_sample = len(x_pool)
    num_train = 100
    num_val = 400
    num_pool = num_sample-num_train-num_val
    
    # Validation samples
    x_val = x_pool[0:num_val]
    y_val = y_pool[0:num_val] 
    
    # Training samples 
    x_train = x_pool[num_val:num_val+num_train]
    y_train = y_pool[num_val:num_val+num_train]
    
    x_pool = x_pool[num_val+num_train:]
    y_pool = y_pool[num_val+num_train:]
    name_pool = name_pool[num_val+num_train:]
    
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
    model.save_weights(os.path.join(r'D:\Faii\Dataset_flood\model', "activelearning-LC-resnet50v2(initial).h5"))
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
        model.load_weights(os.path.join(r'D:\Faii\Dataset_flood\model', "activelearning-LC-resnet50v2(initial).h5"))
        
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
            x=x_train,y=y_train,
            validation_data=(x_val, y_val),
            batch_size=32,
            epochs=num_epochs,
            callbacks=callbacks
        )
        # Store the history of each iteration
        acc.append(history.history['accuracy'])
        val_acc.append(history.history['val_accuracy'])
        loss.append(history.history['loss'])
        val_loss.append(history.history['val_loss'])
          
        # Calculate "U"
        predict_prob = model.predict(x_pool, batch_size=16).flatten()
        u = np.abs(predict_prob-0.5)
        print(u.min())
        
        # Add training samples based on "U"
        sorted_indices = np.argsort(u)
        adding_index = sorted_indices[0:num_add]
        
        # Show least confiences
        img_u = scaler.inverse_transform(x_pool[adding_index].reshape(-1, x_pool[adding_index].shape[-1])).reshape(x_pool[adding_index].shape)
        label_u = y_pool[adding_index]
        name_u = name_pool[adding_index]
      
       
        x_train = np.concatenate((x_train, x_pool[adding_index]), axis=0)
        y_train = np.concatenate((y_train, y_pool[adding_index]), axis=0)
        x_pool = np.delete(x_pool, adding_index, axis=0)
        y_pool = np.delete(y_pool, adding_index, axis=0)
        name_pool = np.delete(name_pool, adding_index, axis=0)

    #%%
    list_val_acc = []  
    list_val_loss = []
    list_train_acc = []  
    list_train_loss = []
    # namefile_loaded =  
    
    for num_iter in range(len(acc)):
        print(num_iter, val_loss[num_iter][-1], val_acc[num_iter][-1])  
        list_val_acc.append(val_acc[num_iter][-1])
        list_val_loss.append(val_loss[num_iter][-1])
        list_train_acc.append(acc[num_iter][-1])
        list_train_loss.append(loss[num_iter][-1])
    plt.figure(figsize=(15,10))
    plt.subplot(2,1,1)
    plt.plot(list_val_loss, label="Validation Loss")
    plt.plot(list_train_loss, label="Train Loss")
    plt.legend()
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(list_val_acc, label="Validation Accuracy")
    plt.plot(list_train_acc, label="Train Accuracy")
    plt.legend()
    plt.grid()
    plt.show()
       
    with io.open('lc_result_val_acc' + str(i),'wb') as f:
       pickle.dump(list_val_acc,f)
 
    with io.open('lc_result_val_loss' + str(i),'wb') as f:
       pickle.dump(list_val_loss,f)
   
    with io.open('lc_result_train_acc' + str(i),'wb') as f:
       pickle.dump(list_train_acc,f)
       
    with io.open('lc_result_train_loss' + str(i),'wb') as f:
      pickle.dump(list_train_loss,f)

        
    ##clearing session
    del model
    del x_train
    del y_train
    del list_val_acc
    del list_val_loss
    K.clear_session()