import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn import preprocessing
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
# import matplotlib.pyplot as plt
from keras import regularizers
import sklearn
# from sklearn import svm
# from sklearn import preprocessing
# from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import time as t

# from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Input

''' which need parallel:
1. classification and regression;
2. parameters: floor in regression;
3. train parameters: [regularizers penalty]
4. train parameters: [dropout]
5. mini-batch graident decent
'''

# read data
def read_data():
    path = 'C:/software/winpython-64bit-3.6.2.0qt5/notebooks/indoor position/long_term_datasets'
    train_rss = pd.read_csv(path + '/train_rss.csv',header  = None)
    train_coord = pd.read_csv(path + '/train_coords.csv', header  = None)

    test_rss = pd.read_csv(path + '/test_rss.csv',header  = None)
    test_coord = pd.read_csv(path + '/test_coords.csv',header  = None)

    return train_rss, train_coord, test_rss, test_coord

#read_data for floors classification
def data_classify():
    train_rss, train_coord, test_rss, test_coord = read_data()
    train_labels = train_coord[-1]
    test_labels = test_coord[-1]
    dummy_label = pd.get_dummies(train_labels)
    train_labels = np.asarray(dummy_label)
    test_labels = np.asarray(pd.get_dummies(test_labels))
    return train_rss, train_labels, test_rss, test_labels

# read data for each floor positioning
def data_regression(floor):
    train_rss, train_coord, test_rss, test_coord = read_data()
    train = pd.concat([train_rss, train_coord], axis=1, ignore_index=True)
    test = pd.concat([test_rss, test_coord], axis=1, ignore_index=True)
    train_replace = train.replace(100,0)
    train_ar = train_replace.values
    test_replace = test.replace(100,0)
    test_ar = test_replace.values

    # read the values in floor #
    train=train_ar[train_ar[:,-1]==floor]
    train_rss = train[:, 0:448]
    train_coord = train[:,448:450]

    test=test_ar[test_ar[:,-1]==floor]
    test_rss = test[0:5000, 0:448]
    test_coord = test[0:5000, 448:450]

    return train_rss, train_coord, test_rss, test_coord

# preprocess the data
def predata(rss, location):
    # the origin of the room
    origin = np.amin(locations,axis=0)
    #size of the room
    room_size = np.amax(locations, axis=0)-origin
    # position respect to origin
    train_Yy = locations - origin
    train_Xx = np.asarray(rss, dtype=np.float64)
    return train_Xx, train_Yy

# split the train data into train and validation data
def train_val(rss, locations):
    train_Xx, train_Yy = predata(rss, locations)
    train_x, val_x, train_y, val_y = train_test_split(train_Xx, train_Yy, test_size=0.25)
    return train_x, val_x, train_y, val_y

# def a fucntion to caculate the accuracy
def accuracy(predictions, labels):
    error = np.sqrt(np.sum((predictions - labels)**2, 1))
    return error, np.mean(error)

# def a function to do positioning
def model_regression(regularzation_penalty, floor):
    rss, locations, test_rss, test_locations = data_regression(floor)
    # get train_X, val_X, train_Y, val_Y
    train_X, val_X, train_Y, val_Y = train_val(rss, locations)
    test_X, test_Y = predata(test_rss, test_locations)

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

    #Model compile
    model.compile(loss='mean_squared_error',optimizer=adam)
    earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=60, verbose=0, mode='auto')
    model.fit(train_X, train_Y,
              epochs=100,
              batch_size=64,callbacks=[earlyStopping],validation_data=(val_X, val_Y))#tbCallBack,
    #model evaluate
    train_loss = model.evaluate(train_X,train_Y, batch_size=len(train_Y)) #calculate the data in test mode(Keras)
    val_loss = model.evaluate(val_X, val_Y, batch_size=len(val_Y))
    test_loss = model.evaluate(test_X, test_Y, batch_size=len(test_Y))
    print("Loss for training data is",train_loss)
    print("Loss for validation data is",val_loss)
    print("Loss for test data is",test_loss)
    predict_Y = model.predict(test_X)
    error_t, accuracy_t = accuracy(predict_Y, test_Y)
    return predict_Y, error_t, accuracy_t

# def a function to do classification
def model_classification(regularzation_penalty):
    # get train_X, val_X, train_Y, val_Y
    rss, floor, test_rss, test_floor = data_classify()
    train_X, val_X, train_Y, val_Y = train_val(rss, floor)

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
    model.add(Dense(1, activation='sigmoid', kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))

    #Model compile
    model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['acc'])
    earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=60, verbose=0, mode='auto')
    model.fit(train_X, train_Y,
              epochs=100,
              batch_size=64,callbacks=[earlyStopping],validation_data=(val_X, val_Y))#tbCallBack,
    #model evaluate
    train_loss, train_accuracy = model.evaluate(train_X,train_Y, batch_size=len(train_Y)) #calculate the data in test mode(Keras)
    val_loss, val_accuracy = model.evaluate(val_X, val_Y, batch_size=len(val_Y))
    test_loss, test_accuracy = model.evaluate(test_X, test_Y, batch_size=len(test_Y))
    print("\n Loss for training data is %f, and the accuracy is %f \n",train_loss, train_accuracy)
    print("\n Loss for validation data is %f,and the accuracy is %f \n",val_loss, val_accuracy)
    print("\n Loss for test data is %f, and the accuracy is %f \n",test_loss,test_accuracy)
    return test_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Text Analysis through TFIDF computation',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', help='Mode of operation',choices=['classification','positioning'])
    parser.add_argument('input', help='Input file or list of files.')
    parser.add_argument('output', help='File in which output is stored')
    parser.add_argument('--floor',default=3,type=int,help ="the floor will train")
    parser.add_argument('--regularzation_penalty',default=0.03,type=float,help ="regularzation_penalty")
    # parser.add_argument('--master',default="local[20]",help="Spark Master")
    # parser.add_argument('--idfvalues',type=str,default="idf", help='File/directory containing IDF values. Used in TFIDF mode to compute TFIDF')
    # parser.add_argument('--other',type=str,help = 'Score to which input score is to be compared. Used in SIM mode')
    args = parser.parse_args()

    if args.mode == "positioning":
        t.time()
        predict_Y, error_t, accuracy_t = model_regression(data_regression(floor = args.floor),args.regularzation_penalty)
        t.time()
        print(accuracy_t)
        np.savetxt('args.output', accuracy_t , delimiter = ',')

    if args.mode == "classification":
        test_accuracy = model_regression(data_classify(),args.regularzation_penalty)
        np.savetxt('args.outpt' , test_accuracy, delimiter = ',')
# positioning_accuracy_t.csv
# classification_accuracy_t.csv