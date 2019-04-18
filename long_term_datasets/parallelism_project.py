import pandas as pd
import numpy as np
import tensorflow as tf
from keras.backend import tensorflow_backend as K
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
# import matplotlib.pyplot as plt
from keras import regularizers
from keras.utils import multi_gpu_model
import sklearn
# from sklearn import svm
# from sklearn import preprocessing
# from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import time as t
import argparse
from sklearn.model_selection import KFold
from statistics import mode, StatisticsError


''' 
wifi indoor positioning project
include floor classification and each floor regression
Training on keras backend tensorflor-gpu

'''

# read data
def read_data():
    test_rss = pd.read_csv('train_rss.csv',header  = None)
    test_coord = pd.read_csv('train_coords.csv', header  = None)

    train_rss = pd.read_csv('test_rss.csv',header  = None)
    train_coord = pd.read_csv('test_coords.csv',header  = None)

    return train_rss, train_coord, test_rss, test_coord

#read_data for floors classification
def data_classify():
    train_rss, train_coord, test_rss, test_coord = read_data()
    train_labels = train_coord.iloc[:, -1]
    test_labels = test_coord.iloc[:, -1]
    dummy_label = pd.get_dummies(train_labels)
    train_labels = np.asarray(dummy_label)
    test_labels = np.asarray(pd.get_dummies(test_labels))
    return train_rss.values, train_labels, test_rss.values, test_labels

# read data for each floor positioning
def data_regression():
    train_rss, train_coord, test_rss, test_coord = read_data()
    train = pd.concat([train_rss, train_coord], axis=1, ignore_index=True)
    test = pd.concat([test_rss, test_coord], axis=1, ignore_index=True)
    train_replace = train.replace(100,0)
    train_ar = train_replace.values
    test_replace = test.replace(100,0)
    test_ar = test_replace.values

    # read the values in floor #
    
    train=train_ar[train_ar[:,450]==3]
    print(train)
    train_rss = train[:, 0:448]
    train_coord = train[:,448:450]
    print(train_rss.shape)
    print(train_coord.shape)

    test=test_ar[test_ar[:,-1]==3]
    test_rss = test[:, 0:448]
    test_coord = test[:, 448:450]

    return train_rss, train_coord, test_rss, test_coord

# preprocess the data
def predata(rss, locations):
    # the origin of the room
    #origin = np.amin(locations,axis=0)
    #size of the room
    #room_size = np.amax(locations, axis=0)-origin
    # position respect to origin
    train_Yy = locations #- origin
    train_Xx = np.asarray(rss, dtype=np.float64)
    return train_Xx, train_Yy

#split the train data into train and validation data, this is simple version of validation, we also use cross validation
def train_val(rss, locations):
    train_Xx, train_Yy = predata(rss, locations)
    train_x, val_x, train_y, val_y = train_test_split(train_Xx, train_Yy, test_size=0.25)
    return train_x, val_x, train_y, val_y

# def a fucntion to caculate the accuracy
def accuracy(predictions, labels):
    error = np.sqrt(np.sum((predictions - labels)**2, 1))
    return error, np.mean(error)


# def a function to do classify accuracy
def class_accuracy(vote_labels, test_labels):
    count = 0
    for i in range(len(vote_labels)):
        if (vote_labels[i] == test_labels[i]).all():
            count = count + 1
    accuracy = 1.*count/len(vote_labels)
    return accuracy

# def a function to get the most likely floor when do classification of floor.
def floor_accuracy(test_label, floor_est):
    vote = np.zeros(len(test_label))
    for i in range(len(test_label)):
         vote[i] = mode(floor_est[i])
    dummy_vote_label = pd.get_dummies(vote)
    vote_labels = np.asarray(dummy_vote_label)
    return class_accuracy(vote_labels, test_label)

# def a function to do positioning
def model_regression(train_X, train_Y, val_X, val_Y,regularzation_penalty, batch):

    # parameters
    num_input = train_X.shape[1]# input layer size
    act_fun = 'relu'
    initilization_method = 'he_normal' #'random_uniform' ,'random_normal','TruncatedNormal' ,'glorot_uniform', 'glorot_nomral', 'he_normal', 'he_uniform'
    #Optimizer
    adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    # define model

    model = Sequential()
    model.add(Dense(128, activation=act_fun, input_dim=num_input, kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation=act_fun, kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
    model.add(Dropout(0.5))
#     model.add(Dense(128, activation=act_fun, kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
#     model.add(Dropout(0.5))
    model.add(Dense(2, activation='linear', kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
    #convert to parallel version
    #parallel_model=multi_gpu_model(model,gpus=1)   

    #Model compile
    model.compile(loss='mean_squared_error',optimizer=adam)
    earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=60, verbose=0, mode='auto')
    model.fit(train_X, train_Y,
              epochs=100,
              batch_size=batch,callbacks=[earlyStopping],validation_data=(val_X, val_Y))#tbCallBack

    return model

# def a function to do classification
def model_classification(train_X, val_X, train_Y, val_Y, regularzation_penalty,batch):

    # parameters
    num_input = train_X.shape[1]# input layer size
    act_fun = 'relu'
    initilization_method = 'he_normal' #'random_uniform' ,'random_normal','TruncatedNormal' ,'glorot_uniform', 'glorot_nomral', 'he_normal', 'he_uniform'
    #Optimizer
    adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    # define model

    model = Sequential()
    model.add(Dense(128, activation=act_fun, input_dim=num_input, kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation=act_fun, kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
    model.add(Dropout(0.5))
#     model.add(Dense(128, activation=act_fun, kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
#     model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid', kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))

    #Model compile
    model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['acc'])
    earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=60, verbose=0, mode='auto')
    model.fit(train_X, train_Y,
              epochs=100,
              batch_size=batch,callbacks=[earlyStopping],validation_data=(val_X, val_Y))#tbCallBack,

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Text Analysis through TFIDF computation',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', help='Mode of operation',choices=['classification','positioning'])
    #parser.add_argument('output', help='Input file or list of files.')
    #parser.add_argument('input', help='File in which output is stored')
    parser.add_argument('--floor',default=3,type=float,help ="the floor will train")
    parser.add_argument('--batch',default=64,type=int,help ="the batch size")
    parser.add_argument('--regularzation_penalty',default=0.03,type=float,help ="regularzation_penalty")
    args = parser.parse_args()

    kf = KFold(n_splits=5, random_state=2018)

    if args.mode == "positioning":
        t1 = t.time()

        rss, locations, test_rss, test_locations = data_regression()
        test_X, test_Y = predata(test_rss, test_locations)
        predict_Y_lst = np.zeros((len(test_X),2,5))

        # get train_X, val_X, train_Y, val_Y by cross validation
        for i, (train_index, test_index) in enumerate(kf.split(rss)):
            print("Fold", i)
            t2 = t.time()
            train_X = rss[train_index]
            train_Y = locations[train_index]
            val_X = rss[test_index]
            val_Y = locations[test_index]

            # with tf.Session(config=tf.ConfigProto(
            #         intra_op_parallelism_threads=32,
            #         inter_op_parallelism_threads=32,
            #         allow_soft_placement=True,
            #         log_device_placement=True)) as sess:
            #     K.set_session(sess)

            model = model_regression(train_X, train_Y, val_X, val_Y, args.regularzation_penalty, args.batch)
            # model evaluate
            train_loss = model.evaluate(train_X, train_Y, batch_size=len(train_Y))  # calculate the data in test mode(Keras)
            val_loss = model.evaluate(val_X, val_Y, batch_size=len(val_Y))
            test_loss = model.evaluate(test_X, test_Y, batch_size=len(test_Y))
            print("Loss for training data is", train_loss)
            print("Loss for validation data is", val_loss)
            print("Loss for test data is", test_loss)
            predict_Y = model.predict(test_X)
            predict_Y_lst[:, :, i] = predict_Y
            error_t, accuracy_t = accuracy(predict_Y, test_Y)
            print('\naccuracy_t:',accuracy_t)
            print('\ntime for each fold:',t.time()-t2)

        pre_Y_mean = np.mean(predict_Y_lst, axis=2)
        error_t, accuracy_t = accuracy(pre_Y_mean, test_Y)

        t3 = t.time()
        print('accuracy_t',accuracy_t,'time', t3-t1)

    if args.mode == "classification":
        t1 = t.time()
        # get train_X, val_X, train_Y, val_Y
        rss, floor, test_rss, test_floor = data_classify()
        floor_est = np.zeros((len(test_rss), 5))
        predict_Y_lst = np.zeros((len(test_rss), 5))

        # get train_X, val_X, train_Y, val_Y by cross validation
        for i, (train_index, test_index) in enumerate(kf.split(rss)):
            print("Fold", i)
            t2 = t.time()
            train_X = rss[train_index]
            train_Y = floor[train_index]
            val_X = rss[test_index]
            val_Y = floor[test_index]

            # with tf.Session(config=tf.ConfigProto(
            #         intra_op_parallelism_threads=8)) as sess:
            #     K.set_session(sess)
            model = model_classification(train_X, val_X, train_Y, val_Y,args.regularzation_penalty,args.batch)
            # model evaluate
            train_loss, train_accuracy = model.evaluate(train_X, train_Y, batch_size=len(
                train_Y))  # calculate the data in test mode(Keras)
            val_loss, val_accuracy = model.evaluate(val_X, val_Y, batch_size=len(val_Y))
            test_loss, test_accuracy = model.evaluate(test_rss, test_floor, batch_size=len(test_floor))
            floor_est[:, i] = model.predict_classes(test_rss)
            print("\n Loss for training data is %f, and the accuracy is %f \n", train_loss, train_accuracy)
            print("\n Loss for validation data is %f,and the accuracy is %f \n", val_loss, val_accuracy)
            print("\n Loss for test data is %f, and the accuracy is %f \n", test_loss, test_accuracy)
            print('\ntime for each fold:',t.time()-t2)

        cv_test_accuracy = floor_accuracy(test_floor, floor_est)

        t3 = t.time()
        print('time',t3-t1, '\n cv_test_accuracy',cv_test_accuracy)

