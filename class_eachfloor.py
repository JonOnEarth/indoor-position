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
import auto_regression as ar
import regular_regression as rr
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.model_selection import KFold

class each_floor():
    def __init__(self, train_r, train_c, test_r, test_c):
        self.train_r=train_r
        self.train_c=train_c
        self.test_r=test_r
        self.test_c=test_c
    
    def predata(rss, locations):
        # the origin of the room
        origin = np.amin(locations,axis=0)
        #size of the room
        room_size = np.amax(locations, axis=0)-origin
        # position respect to origin
        train_Yy = locations - origin
        train_Xx = np.asarray(rss, dtype=np.float64)
        return train_Xx, train_Yy


    def accuracy(predictions, labels):
        error = np.sqrt(np.sum((predictions - labels)**2, 1))
        return error, np.mean(error)

    def t_split(self):
        trainX, trainY = predata(self.train_r, self.train_c)
        testX, testY = predata(self.test_r, self.test_c)
        return trainX,trainY,testX,testY
    
    def reshaped(predict):
        size = predict.shape[0]
        j = predict.reshape((2*size, 1))
        return j
    
    
   
    

   # function that creat the dataset for second layer model
    def get_oof(self,clf):
#     X_train = reshaped(X_train)
#     y_train = reshaped(y_train)
#     X_test = reshaped(X_test)
        X_train,y_train,X_test, y_test = t_split(self)
        
    
        
        blend_train = np.zeros((y_train.shape[0],2))
        blend_test = np.zeros((X_test.shape[0],2))
        blend_test_skf = np.zeros((X_test.shape[0],2,5)) 

        kf = KFold(n_splits=5, random_state=2018)
        
        # do the cross-validation
        for i, (train_index, test_index) in enumerate(list(kf.split(X_train))):
            print("Fold", i)   

            kf_X_train = X_train[train_index]
            kf_y_train = y_train[train_index]
            kf_X_test = X_train[test_index]
            kf_y_test = y_train[test_index]

            model = clf(kf_X_train,kf_y_train)

            blend_train[test_index]=model.predict(kf_X_test)  # 992*2

            blend_test_skf[:,:,i] = model.predict(X_test)   # 1*292*2

        blend_test[:,:]=blend_test_skf.mean(axis=2)
        return blend_train, blend_test
    
    ##
    def blend_train(self):
        clf1=rr.train_model
        clf2=ar.regression

        neigh = KNeighborsRegressor(n_neighbors=4)
        clf3 = neigh.fit

        clf4 = rr.train_model2

        #clf5 = rr.train_model3

        dr = DecisionTreeRegressor(max_depth = 9)
        clf5 = dr.fit

        clfs = [clf1,clf2,clf3,clf4,clf5]
    
        blend_train1, blend_test1 = get_oof(self, clf1)
        e1, e1_mean = accuracy(blend_test1, testY)
        print("el_mean: ", e1_mean)
        blend_train2, blend_test2 = get_oof(self, clf2)
        e2, e2_mean = accuracy(blend_test2, testY)
        print("e2_mean: ", e2_mean)
        blend_train3, blend_test3 = get_oof(self, clf3)
        e3, e3_mean = accuracy(blend_test3, testY)
        print("e3_mean: ", e3_mean)
        blend_train4, blend_test4 = get_oof(self, clf4)
        e4, e4_mean = accuracy(blend_test4, testY)
        print("e4_mean", e4_mean)
        blend_train5, blend_test5 = get_oof(self, clf5)
        e5, e5_mean = accuracy(blend_test5, testY)
        print("e5_mean: ", e5_mean)

        ## 
        blend_trainX = np.hstack((blend_train1,blend_train2, blend_train3,blend_train4,blend_train5))
        blend_testX = np.hstack((blend_test1, blend_test2, blend_test3,blend_test4,blend_test5))

        model_stack = Sequential()
        model_stack.add(Dense(32, input_dim=10, activation='linear', bias=True))
        model_stack.add(Dense(16, activation='linear', bias=True))
        model_stack.add(Dense(2, activation='linear', bias=True))
        model_stack.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
        model_stack.fit(blend_trainX,trainY,nb_epoch=50)
        predict_1 = model_stack.predict(blend_testX)
        e_stack, e_stack_mean = accuracy(predict_1, testY)
        print(e_stack_mean)

        print('minimum error:', np.amin(e_stack), 'maximum error:', np.amax(e_stack), 'variance:', np.var(e_stack),"median: ", np.median(e_stack))

        error_sorted= np.sort(e_stack)
        p = 1. *np.arange(len(e_stack))/(len(e_stack)-1)
        plt.plot(error_sorted, p)
        plt.show()


