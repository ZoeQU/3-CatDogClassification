#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import Sequential

model = tf.keras.models.load_model('model_catsVSdogs_10epoch.h5')

"""define image properties"""
Image_Width = 128
Image_Height = 128
Image_Size = (Image_Width,Image_Height)
Image_Channel = 3

"""test model"""
results = {0:'cat', 1:'dog'}
im = Image.open("image_path")
im = im.resize(Image_Size)
im = np.expand_dimsnd(im,axis=0)
im = np.array(im)
im = im/255
pred = model.predict_classes([im])[0]
print(pred, results[pred])