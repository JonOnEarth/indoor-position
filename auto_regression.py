import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn import preprocessing
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
import matplotlib.pyplot as plt
from keras import regularizers
import sklearn
from sklearn import svm
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

nb_epochs = 100
batch_size = 64
input_size = 992
num_classes = 2

def encoder():
    model = Sequential()
    model.add(Dense(512, input_dim=input_size, activation='relu', bias=True))
    model.add(Dense(256, activation='relu', bias=True))
    #model.add(Dense(128, activation='relu', bias=True))
    return model

def decoder(e):   
    #e.add(Dense(256, input_dim=128, activation='relu', bias=True))
    e.add(Dense(512, input_dim=256, activation='relu', bias=True))
    e.add(Dense(input_size, activation='relu', bias=True))
    e.compile(optimizer='adam', loss='mse')
    return e

def train_val(rss, locations):
    
    train_x, val_x, train_y, val_y = train_test_split(rss, locations, test_size=0.2)
    return train_x, val_x, train_y, val_y


def regression(rss, locations):
    train_X, val_X, train_Y, val_Y = train_val(rss, locations)
    e = encoder()
    d = decoder(e)
    d.fit(train_X, train_X, nb_epoch=nb_epochs, batch_size=batch_size)
    num_to_remove = 2
    regularzation_penalty = 0.02
    initilization_method = 'he_normal' #'random_uniform' ,'random_normal','TruncatedNormal' ,'glorot_uniform', 'glorot_nomral', 'he_normal', 'he_uniform'
    #Optimizer
    adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    for i in range(num_to_remove):
        d.pop()
    d.add(Dense(256, input_dim=256, activation='relu', kernel_initializer=initilization_method, kernel_regularizer=regularizers.l2(regularzation_penalty)))
    d.add(Dropout(0.5))
    d.add(Dense(256, activation='relu', kernel_initializer=initilization_method, kernel_regularizer=regularizers.l2(regularzation_penalty)))
    d.add(Dropout(0.5))
    d.add(Dense(num_classes, activation='linear', kernel_initializer=initilization_method, kernel_regularizer=regularizers.l2(regularzation_penalty)))

    #Model compile
    d.compile(loss='mean_squared_error',
                  optimizer='adam')
    
    earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=60, verbose=0, mode='auto')
    Model_best= keras.callbacks.ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)

    d.fit(train_X, train_Y, validation_data=(val_X, val_Y), nb_epoch=nb_epochs, callbacks=[earlyStopping, Model_best], batch_size=batch_size)
    return d