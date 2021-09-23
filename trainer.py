#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.applications.imagenet_utils import (decode_predictions,preprocess_input)
from keras.layers import (Conv2D, Dense, Flatten, GlobalAveragePooling2D,GlobalMaxPool2D,Input,Dropout,MaxPooling2D,BatchNormalization)
from keras.preprocessing import image
from keras import models,Sequential
from keras.utils import plot_model


"""define image properties"""
Image_Width = 128
Image_Height = 128
Image_Size = (Image_Width,Image_Height)
Image_Channel = 3

"""prepare training dataset"""
filenames = os.listdir('./dogs-vs-cats/train')

categories = []
for f_name in filenames:
    category = f_name.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename':filenames,
    'category':categories
})
"""creat the NN model"""
model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(Image_Width,Image_Height,Image_Channel)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25)) #dropout 25% data

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25)) #dropout 25% data

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25)) #dropout 25% data

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5)) #dropout 50% data
model.add(Dense(2,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()
plot_model(model, "net.svg", show_shapes=True)
"""define callbacks and learning rate"""
earlystop = tf.keras.callbacks.EarlyStopping(patience=10)
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc',patience=2,verbose=1,factor=0.5,min_lr=0.00001)
callbacks = [earlystop,learning_rate_reduction]

"""manage data"""
df['category']=df['category'].replace({0:'cat',1:'dog'})
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size = 15

"""training and validation data generator"""
"""data augmentation"""
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15,
                                                               rescale=1./255,
                                                               shear_range=0.1,
                                                                zoom_range=0.2,
                                                                horizontal_flip=True,
                                                                width_shift_range=0.1,
                                                                height_shift_range=0.1)
train_generator = train_datagen.flow_from_dataframe(train_df,'./dogs-vs-cats/train/',
                                                    x_col='filename',
                                                    y_col='category',
                                                    target_size=Image_Size,
                                                    class_mode='categorical',
                                                    batch_size=batch_size)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(validate_df,'./dogs-vs-cats/train/',
                                                    x_col='filename',
                                                    y_col='category',
                                                    target_size=Image_Size,
                                                    class_mode='categorical',
                                                    batch_size=batch_size)

"""model training"""
# epochs = 10
# history = model.fit_generator(train_generator,
#                               epochs=epochs,
#                               validation_data=validation_generator,
#                               validation_steps=total_validate//batch_size,
#                               steps_per_epoch=total_train//batch_size,
#                               callbacks=callbacks)

"""save the model"""
model.save('model_catsVSdogs_10epoch.h5')

"""test data preparation"""
model = tf.keras.models.load_model('model_catsVSdogs_10epoch.h5')
test_filenames = os.listdir('./dogs-vs-cats/test1')
test_df = pd.DataFrame({'filename':test_filenames})

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15,
                                                               rescale=1./255,
                                                               shear_range=0.1,
                                                                zoom_range=0.2,
                                                                horizontal_flip=True,
                                                                width_shift_range=0.1,
                                                                height_shift_range=0.1)
test_generator = test_datagen.flow_from_dataframe(test_df,'./dogs-vs-cats/test1/',
                                                    x_col='filename',
                                                    y_col='category',
                                                    target_size=Image_Size,
                                                    class_mode=None,
                                                    batch_size=batch_size)

nb_samples = test_df.shape[0]
predict = model.predict_generator(test_generator,steps=np.ceil(nb_samples/batch_size))
test_df['category']=np.argmax(predict, axis=-1)
# label_map = dict((v,k) for k,v in train_generator.class_indices.items())
label_map = (train_generator.class_indices)
label_map = dict((v,k) for k,v in label_map.items()) #flip k,v
test_df['category'] = test_df['category'].replace(label_map)
test_df['category'] = test_df['category'].replace({'dog':1,'cat':0})

"""visualize the prediction results"""
sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12,24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = tf.keras.preprocessing.image.load_img('./dogs-vs-cats/test1/'+filename,
                                                target_size=Image_Size)
    plt.subplot(6,3,index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + '{}'.format(category) + ')')
plt.tight_layout()
plt.show()



