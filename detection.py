# -*- coding: utf-8 -*-
from math import cos, sin, pi, degrees

from skimage.transform import resize, rotate

from keras.models import Sequential
from keras.layers import Dense, SpatialDropout2D, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adadelta
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping
from keras import backend as K

import numpy as np
from numpy.random import uniform as rnd
import matplotlib.pyplot as plt

FACEPOINTS_COUNT = 14

REG = 0.0
IMAGE_SIZE = 64
VAL_SIZE = 1000
BATCH_SIZE = 64
DROP_OUT = 0.3
N_EPOCH = 25
PATIENCE = 8
INIT = 'he_normal'
ReLU = 1./5.5

N_FILTERS1 = 32   # CONV1
N_FILTERS2 = 128  # CONV2
N_FILTERS3 = 256  # CONV3
N_FILTERS4 = 512  # FC

#DATA AUGMETATION
ARG = pi/30
SCALE = 0.8
P = 0.55

# print('===========================')
# print('REG: %f' % REG)
# print('DROP_OUT: %f' % DROP_OUT)
# print('INIT: %s' % INIT)
# print('N_EPOCH: %d' % N_EPOCH)
# print('ReLU: %f' % ReLU)
# print('PATIENCE: %d' % PATIENCE)
# print('===========================')
# print('IMAGE_SIZE: %d' % IMAGE_SIZE)
# print('VAL_SIZE: %d' % VAL_SIZE)
# print('BATCH_SIZE: %d' % BATCH_SIZE)
# print('===========================')
# print('N_FILTERS1: %d' % N_FILTERS1)
# print('N_FILTERS2: %d' % N_FILTERS2)
# print('N_FILTERS3: %d' % N_FILTERS3)
# print('N_FILTERS4: %d' % N_FILTERS4)
# print('===========================')

def flip_y(y_data):
    flip_indices = [(0, 3), (1, 2), (4, 9),
                    (5, 8), (6, 7), (11, 13)]
    y = y_data.copy()
    y[:,0] = -y[:,0]
    for i, j in flip_indices:
        y[i,:], y[j,:] = (y[j,:].copy(), y[i,:].copy())
    return y

def rotate_y(y_data, arg):
    y = y_data.copy()
    y[:,0] = cos(arg) * y[:,0] + sin(arg) * y[:,1]  # cos(a)  sin(a)
    y[:,1] = cos(arg) * y[:,1] - sin(arg) * y[:,0]  # -sin(a)  cos(a) clockwize
    return y

def contrast(X_data, scale):
    X = X_data.copy()
    X[...,0] = scale * X[...,0] + (1 - scale) * np.mean(X[..., 0])
    X[...,1] = scale * X[...,1] + (1 - scale) * np.mean(X[..., 1])
    X[...,2] = scale * X[...,2] + (1 - scale) * np.mean(X[..., 2])
    return X

class BarNet:
    
    def __init__(self):
        self.model = Sequential()
        
        #CONV1
        self.model.add(Convolution2D(N_FILTERS1, 3, 3, init=INIT, input_shape=(3, IMAGE_SIZE, IMAGE_SIZE)))
        self.model.add(LeakyReLU(alpha=ReLU))
        
        #MAX_POOL1
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        self.model.add(BatchNormalization())
        
        #CONV2
        self.model.add(Convolution2D(N_FILTERS2, 2, 2, init=INIT))
        self.model.add(LeakyReLU(alpha=ReLU))
        self.model.add(SpatialDropout2D(0.1))

        #MAX_POOL2
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        self.model.add(BatchNormalization())
        
        #CONV3
        self.model.add(Convolution2D(N_FILTERS3, 2, 2, init=INIT))
        self.model.add(LeakyReLU(alpha=ReLU))
        self.model.add(SpatialDropout2D(0.2))

        #MAX_POOL3
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        self.model.add(BatchNormalization())

        #FC1
        self.model.add(Flatten())
        self.model.add(Dense(N_FILTERS4, init=INIT))
        self.model.add(LeakyReLU(alpha=ReLU))
        self.model.add(Dropout(DROP_OUT))
        
        #FC2
        self.model.add(Dense(N_FILTERS4, init=INIT))
        self.model.add(LeakyReLU(alpha=ReLU))
        self.model.add(Dropout(DROP_OUT))

        #FC3
        self.model.add(Dense(N_FILTERS4, init=INIT))
        self.model.add(LeakyReLU(alpha=ReLU))

        #KEY_POINTS
        self.model.add(Dense(2*FACEPOINTS_COUNT, init='he_uniform'))
        self.model.add(Reshape((FACEPOINTS_COUNT, 2), input_shape=(2*FACEPOINTS_COUNT, )))

    def set_data(self, X, y=None):
        if y is None:
            self.X_test = X
        else:
            train_size = len(X) - VAL_SIZE
            mask = range(train_size)
            self.X_train = X[mask]
            self.y_train = y[mask]
        
            mask = range(train_size, train_size + VAL_SIZE)
            self.X_val = X[mask]
            self.y_val = y[mask]

    def set_mean_image(self):
        self.mean_image = []
        self.mean_image.append(np.mean(self.X_train[..., 0], axis=0))
        self.mean_image.append(np.mean(self.X_train[..., 1], axis=0))
        self.mean_image.append(np.mean(self.X_train[..., 2], axis=0))
    
    def zero_center(self, mode):
        if mode == 'train':
            self.X_train[...,0] -= self.mean_image[0]
            self.X_train[...,1] -= self.mean_image[1]
            self.X_train[...,2] -= self.mean_image[2]
            
            self.X_val[...,0] -= self.mean_image[0]
            self.X_val[...,1] -= self.mean_image[1]
            self.X_val[...,2] -= self.mean_image[2]
        elif mode == 'test':
            self.X_test[...,0] -= self.mean_image[0]
            self.X_test[...,1] -= self.mean_image[1]
            self.X_test[...,2] -= self.mean_image[2]

    def augment_data(self):
        flip_indices = [(0, 3), (1, 2), (4, 9),
                        (5, 8), (6, 7), (11, 13)]
        X_aug, y_aug = ([],[])
        
        for i in range(len(self.X_train)):
            new_X = [self.X_train[i]]
            new_y = [self.y_train[i]]
            
            if rnd(0, 1) < P:
                new_X.append(new_X[0][:,::-1,:])
                new_y.append(flip_y(new_y[0]))
            
            #Contrast NEW
            for j in range(len(new_X)):
                if rnd(0, 1) < P/2:
                    new_X.append(contrast(new_X[j], SCALE))
                    new_y.append(new_y[j])

            for j in range(len(new_X)):
                if rnd(0, 1) < P+0.05:
                    sgn = np.sign(rnd(-1, 1))
                    arg = sgn * rnd(ARG, 2*ARG)
                    new_X.append(rotate(new_X[j], degrees(arg)))
                    new_y.append(rotate_y(new_y[j], arg))

                #Rotation 90 NEW
                if rnd(0, 1) < P+0.05:
                    sgn = np.sign(rnd(-1, 1))
                    arg = sgn * (pi/2 - rnd(ARG, 2*ARG))
                    new_X.append(rotate(new_X[j], degrees(arg)))
                    new_y.append(rotate_y(new_y[j], arg))

            X_aug += new_X
            y_aug += new_y
    
        self.X_train = np.array(X_aug)
        self.y_train = np.array(y_aug)

    def train(self):
        self.X_train = self.X_train.transpose(0, 3, 1, 2)
        self.X_val = self.X_val.transpose(0, 3, 1, 2)

        self.model.compile(optimizer=Adadelta(), loss='mse')
        self.model.fit(self.X_train, self.y_train, batch_size=BATCH_SIZE, callbacks=[EarlyStopping(patience=PATIENCE)],
                nb_epoch=N_EPOCH, validation_data=(self.X_val, self.y_val))
    
    def predict(self):
        self.X_test = self.X_test.transpose(0, 3, 1, 2)
        result = self.model.predict(self.X_test, batch_size=BATCH_SIZE)
        return result.reshape(len(self.X_test), FACEPOINTS_COUNT, 2)


def resize_images(X):
    X = list(X)
    ratios = np.ones((len(X), FACEPOINTS_COUNT, 2), dtype=np.float64)
    
    for i in range(len(X)):
        ratios[i] *= 2. / X[i].shape[0]
        X[i] = resize(X[i], (IMAGE_SIZE, IMAGE_SIZE, 3))

    return [np.array(X), ratios]

def train_detector(X, y):
    model = BarNet()
    print("Initialization is completed")

    X, ratios = resize_images(X)
    y = y * ratios - 1
    print("Resizing is completed")
    model.set_data(X, y)
    model.augment_data()
    print("Data augmentation is completed")

    model.set_mean_image()
    model.zero_center('train')
    print("Preprocessing is completed")
    
    model.train()
    
    return model

def detect(model, X):
    X, ratios = resize_images(X)
    
    X_pred = []
    for i in range(len(X)):
        X_pred.append(X[i])
        X_pred.append(X[i,:,::-1,:])
        #Contrast NEW
        X_pred.append(contrast(X[i], SCALE))
    
    X_pred = np.array(X_pred)
    model.set_data(X_pred)
    model.zero_center('test')
    y = model.predict()
    
    y_pred = []
    step = 3
    for i in range(0, len(y), step):
        y[i+1] = flip_y(y[i+1])
        mean_y = ((y[i]+y[i+1]+y[i+2])/step + 1)/ratios[i//step]
        y_pred.append(np.round(mean_y))
    
    return y_pred
