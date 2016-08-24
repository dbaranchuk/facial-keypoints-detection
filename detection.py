# -*- coding: utf-8 -*-

from math import cos, sin, pi, degrees

from skimage.transform import resize, rotate

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, Callback
from keras import backend as K

#from face_detector import FaceDetector

import numpy as np
import matplotlib.pyplot as plt

FACEPOINTS_COUNT = 14

reg = 0.0
lr_rt = 9e-03
lr_dc = 5e-04

REG = l2(reg)

IMAGE_SIZE = 128
VAL_SIZE = 0.10
BATCH_SIZE = 64
DROP_OUT = 0.25
N_EPOCH = 50
PATIENCE = 50
INIT = 'he_normal'
ReLU = 0.01
MOMENTUM = 0.9

N_FILTERS1 = 32  # 96 64    # CONV1
N_FILTERS2 = 64  # 128 96   # CONV2
N_FILTERS3 = 128 # 192 128  # CONV3
N_FILTERS4 = 256 # 256 192  # CONV4
N_FILTERS5 = 512 # 512 256  # FC

print('===========================')
print('NEW ARCHITECTURE')
print('MAX_POOL = MaxPooling2D')
print('===========================')
print(reg, lr_rt, lr_dc)
print('===========================')
print('MOMENTUM: %f' % MOMENTUM)
print('DROP_OUT: %f' % DROP_OUT)
print('INIT: %s' % INIT)
print('N_EPOCH: %d' % N_EPOCH)
print('ReLU: %f' % ReLU)
print('PATIENCE: %d' % PATIENCE)
print('===========================')
print('IMAGE_SIZE: %d' % IMAGE_SIZE)
print('VAL_SIZE: %f' % VAL_SIZE)
print('BATCH_SIZE: %d' % BATCH_SIZE)
print('===========================')
print('N_FILTERS1: %d' % N_FILTERS1)
print('N_FILTERS2: %d' % N_FILTERS2)
print('N_FILTERS3: %d' % N_FILTERS3)
print('N_FILTERS4: %d' % N_FILTERS4)
print('N_FILTERS5: %d' % N_FILTERS5)
print('===========================')

class LearningRateDecay(Callback):
    def __init__(self):
        self.prev_val_loss = 1.
        self.flag = False
        self.decay = 0.001

    def on_epoch_end(self, epoch, logs={}):
        if self.model.optimizer.lr.eval() < self.decay:
            self.decay /= 100
        
        cur_val_loss = round(logs.get('val_loss'), 4)
        if cur_val_loss >= self.prev_val_loss:
            if self.flag:
                self.model.optimizer.lr -= self.decay
                optimizer = self.model.optimizer
                lr = (optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations))).eval()
                print('\nLR: {:.8f}'.format(float(lr)))
                self.flag = False
            else:
                self.flag = True

        self.prev_val_loss = cur_val_loss

class BarNet:
    
    def __init__(self):
        self.model = Sequential()

        #CONV1
        self.model.add(Convolution2D(N_FILTERS1, 3, 3, init=INIT, input_shape=(3, IMAGE_SIZE, IMAGE_SIZE))) #W_regularizer=REG,
        self.model.add(LeakyReLU(alpha=ReLU))
        
        #MAX_POOL1
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        self.model.add(BatchNormalization())
        
        #CONV2
        self.model.add(Convolution2D(N_FILTERS2, 2, 2, init=INIT))#, W_regularizer=REG))
        self.model.add(LeakyReLU(alpha=ReLU))

        #MAX_POOL2
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        self.model.add(BatchNormalization())
        
        #CONV3
        self.model.add(Convolution2D(N_FILTERS3, 2, 2, init=INIT))#, W_regularizer=REG))
        self.model.add(LeakyReLU(alpha=ReLU))

        #MAX_POOL3
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        self.model.add(BatchNormalization())

        #CONV4
        self.model.add(Convolution2D(N_FILTERS4, 2, 2, init=INIT))#, W_regularizer=REG))
        self.model.add(LeakyReLU(alpha=ReLU))

        #MAX_POOL4
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        self.model.add(BatchNormalization())

        #FC1
        self.model.add(Flatten())
        self.model.add(Dense(N_FILTERS5, init=INIT))#W_regularizer=REG))
        self.model.add(LeakyReLU(alpha=ReLU))
        self.model.add(Dropout(DROP_OUT))
        
        #FC2
        self.model.add(Dense(N_FILTERS5, init=INIT))# W_regularizer=REG))
        self.model.add(LeakyReLU(alpha=ReLU))
#        self.model.add(Dropout(DROP_OUT))

        #KEY_POINTS
        self.model.add(Dense(2*FACEPOINTS_COUNT, init='he_uniform'))
        self.model.add(Reshape((FACEPOINTS_COUNT, 2), input_shape=(2*FACEPOINTS_COUNT, )))

    def train(self, X_train, y_train):
        X_train = X_train.transpose(0, 3, 1, 2)
        
        sgd = SGD(lr=lr_rt, decay=lr_dc, momentum=MOMENTUM, nesterov=True)
        
        self.model.compile(optimizer=sgd, loss='mse')
        self.model.fit(X_train, y_train, batch_size=BATCH_SIZE,
                       callbacks=[EarlyStopping(patience=PATIENCE), LearningRateDecay()],
                       nb_epoch=N_EPOCH, validation_split=VAL_SIZE)

    def predict(self, X_test):
        X_test = X_test.transpose(0, 3, 1, 2)
        result = self.model.predict(X_test, batch_size=BATCH_SIZE)
        return result.reshape(len(X_test), FACEPOINTS_COUNT, 2)

#Preprocessings
def resize_images(X):
    X = list(X)
    ratios = np.ones((len(X), FACEPOINTS_COUNT, 2), dtype=np.float64)
    
    for i in range(len(X)):
        ratios[i] *= 2. / X[i].shape[0]
        X[i] = resize(X[i], (IMAGE_SIZE, IMAGE_SIZE, 3))

    return [np.array(X), ratios]

def augment_data(X, y, alpha=0, size=2):
    indices = np.random.choice(len(X), len(X)//size)
    
    new_X = X[indices]
    new_y = y[indices]

    if alpha == 0:
        flip_indices = [(0, 3), (1, 2), (4, 9),
                        (5, 8), (6, 7), (11, 13)]
        new_X = new_X[:,:,::-1,:]
        new_y[:,:,0] = -1. * new_y[:,:,0]
    
        for i, j in flip_indices:
            new_y[:, i], new_y[:, j] = (new_y[:, j].copy(), new_y[:, i].copy())
            
    else:
        for i in range(len(new_X)):
            r = np.random.uniform(0, 1)
            arg = 0
            
            if r < 0.5:
                arg = alpha
            else:
                arg = -alpha
            
            new_y[i,:,0] = cos(arg) * new_y[i,:,0] + sin(arg) * new_y[i,:,1]  # cos(a)  sin(a)
            new_y[i,:,1] = cos(arg) * new_y[i,:,1] - sin(arg) * new_y[i,:,0]  # -sin(a) cos(a)
            new_X[i] = rotate(new_X[i], degrees(arg))

    return [new_X, new_y]

def train_detector(X, y):
    
#    face_detector = FaceDetector()
#    face_detector.set_data(X, y)
#    face_detector.train()
#    print("FaceDetector is trained")
    
    model = BarNet()
    print("Initialization is completed")
    
    #preprocessing
    X, ratios = resize_images(X)
    y = y * ratios - 1
    print("Preprocessing is completed")
    
    #Data Augmentation
    flip_X, flip_y = augment_data(X, y)
    print("Flipping augmentation is completed")

    rot10_X, rot10_y = augment_data(X, y, alpha=(pi/18))
    print("Rotation 10 augmentation is completed")
  
#    rot90_X, rot90_y = augment_data(X, y, alpha=pi/2, size=4)
#    print("Rotation 90 augmentation is completed")

    X = np.vstack((X, flip_X, rot10_X))
    y = np.vstack((y, flip_y, rot10_y))

    X[..., 0] -= np.mean(X[..., 0], axis=0)
    X[..., 1] -= np.mean(X[..., 1], axis=0)
    X[..., 2] -= np.mean(X[..., 2], axis=0)

    model.train(X, y)

    return model


def detect(model, X):
    #preprocessing
    X, ratios = resize_images(X)

    X[..., 0] -= np.mean(X[..., 0], axis=0)
    X[..., 1] -= np.mean(X[..., 1], axis=0)
    X[..., 2] -= np.mean(X[..., 2], axis=0)
    
    y_pred = np.round((model.predict(X) + 1) / ratios)
    return list(y_pred)

