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



def train_val(rss, locations):
    train_x, val_x, train_y, val_y = train_test_split(rss, locations, test_size=0.2)
    return train_x, val_x, train_y, val_y


def train_model(rss, locations):
    
    train_X, val_X, train_Y, val_Y = train_val(rss, locations)

    # parameters
    num_input = train_X.shape[1]# input layer size
    act_fun = 'relu'
    regularzation_penalty = 0.03
    initilization_method = 'he_normal' #'random_uniform' ,'random_normal','TruncatedNormal' ,'glorot_uniform', 'glorot_nomral', 'he_normal', 'he_uniform'
    #Optimizer
    adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    # define model

    model = Sequential()
    model.add(Dense(512, activation=act_fun, input_dim=num_input, kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation=act_fun, kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
    model.add(Dropout(0.5))
#     model.add(Dense(128, activation=act_fun, kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
#     model.add(Dropout(0.5))
    model.add(Dense(2, activation='linear', kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))

    #Model compile
    model.compile(loss='mean_squared_error',
                  optimizer=adam)
    earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=60, verbose=0, mode='auto')
    model.fit(train_X, train_Y,
              epochs=100,
              batch_size=64,callbacks=[earlyStopping],validation_data=(val_X, val_Y))#tbCallBack,
    
    return model

def train_model2(rss, locations):
    train_X, val_X, train_Y, val_Y = train_val(rss, locations)

    # parameters
    num_input = train_X.shape[1]# input layer size
    
    # define model

    model = Sequential()
    model.add(Dense(128, input_dim=992, activation='relu', bias=True))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', bias=True))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='linear', bias=True))
    model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
    earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=60, verbose=0, mode='auto')
    model.fit(train_X, train_Y,
              epochs=100,
              batch_size=64,callbacks=[earlyStopping],validation_data=(val_X, val_Y))#tbCallBack,
    
    return model

def train_model3(rss, locations):
    train_X, val_X, train_Y, val_Y = train_val(rss, locations)
    
    # parameters
    num_input = train_X.shape[1]# input layer size
    
    model = Sequential()
    model.add(Dense(256, input_dim=992, activation='relu', bias=True))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', bias=True))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='linear', bias=True))
    model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
    earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=60, verbose=0, mode='auto')
    model.fit(train_X, train_Y,
              epochs=100,
              batch_size=64,callbacks=[earlyStopping],validation_data=(val_X, val_Y))#tbCallBack,
    
    return model
    

