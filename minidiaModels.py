#-----------------------------------------
#Import neccesary library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import tensorflow as tf
import math
import sklearn
import scikeras
from tensorflow import keras
from keras.models import Sequential
#from tensorflow.keras.layers import Dense, Activation, Dropout
from keras.layers import Dense, Activation, Dropout
from sklearn.preprocessing import MinMaxScaler
#from tensorflow.keras.constraints import MaxNorm
from keras.constraints import MaxNorm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from keras.layers import ReLU
from keras.layers import LSTM, Flatten
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor
import datetime


#===============================================
#define the grid search parameters -- parameter_grid = dict(neurons=neurons, dropout_rate = dropout_rate, batch_size=batchSize, epochs=epochs)
parameter_grid_LSTM = {'model__neurons': [2, 10, 32, 64, 128, 200],
                  'batch_size': [50, 100, 200, 300],
                  'epochs': [10, 20, 30, 50, 100]
}


#-----------------------------------------
#Build LSTM model...
#Input: ...
#Output: ...
def buildLSTM(trainX, trainY, cv, seq_size):

    def my_LSTM_model(neurons = 10, batch_size = 50, epochs = 20):
        __my_model = Sequential()
        __my_model.add(LSTM(neurons, input_dim = seq_size, kernel_constraint= MaxNorm(1), recurrent_constraint = MaxNorm(1), activation='relu'))
        __my_model.add(Dropout(0.01))
        __my_model.add(Dense(neurons, activation='relu'))
        __my_model.add(Dense(1, activation='relu'))
        __my_model.compile(loss='mean_squared_error', optimizer= 'Adam', metrics=['mae'])
        return __my_model
    
    my_model = KerasRegressor(model = my_LSTM_model, epochs = 20, verbose=1)#, input_dim = seq_size)

    grid = GridSearchCV(estimator=my_model, param_grid=parameter_grid_LSTM, n_jobs=-1, cv=cv)

    grid_result = grid.fit(trainX, trainY)

    best_model = grid_result.best_estimator_

    return best_model


#===============================================

#define the grid search parameters -- parameter_grid = dict(neurons=neurons, dropout_rate = dropout_rate, batch_size=batchSize, epochs=epochs)
parameter_grid_MLP = {'model__neurons': [2, 10, 32, 64, 128, 200],
                  'batch_size': [50, 100, 200, 300],
                  'epochs': [10, 20, 30, 50, 100]
}


#-----------------------------------------
#Build LSTM model...
#Input: ...
#Output: ...
def buildMLP(trainX, trainY, cv, seq_size):

    def my_MLP_model(neurons = 10, batch_size = 50, epochs = 20):
        my_model = Sequential()
        my_model.add(Dense(neurons, input_dim = (seq_size), kernel_constraint= MaxNorm(1), activation='relu'))
        my_model.add(Dropout(0.01))
        my_model.add(Dense(neurons, activation='relu'))
        my_model.add(Dense(1, activation='relu'))
        my_model.compile(loss='mean_squared_error', optimizer= 'Adam', metrics=['mae'])
        return my_model
    
    my_model = KerasRegressor(model = my_MLP_model, epochs = 20, verbose=1)

    grid = GridSearchCV(estimator=my_model, param_grid=parameter_grid_LSTM, n_jobs=-1, cv=cv)

    grid_result = grid.fit(trainX, trainY)

    best_model = grid_result.best_estimator_

    return best_model



#====================================================

#-----------------------------------------
#Model selection and bulding...
#Input: ...
#Output: ...
def buildModel(trainX, trainY, cvFolds, modelOpt, seq_size):
    if modelOpt == 'lstm':
        model = buildLSTM(trainX, trainY, cvFolds, seq_size)

    if modelOpt == 'mlp':
        model = buildMLP(trainX, trainY, cvFolds, seq_size)

    return model



#-----------------------------------------
#Model summary...
#Input: ...
#Output: ...
def modelSummary(trainX, trainY, testX, testY, scaler, best_model, pathReport, modelOpt):

    print('', file=open(pathReport, 'a'))
    print(':: %s process :: %s ' % (modelOpt, datetime.datetime.now()), file=open(pathReport, 'a'))
    print('', file=open(pathReport, 'a'))

    train_pred = best_model.predict(trainX)
    test_pred = best_model.predict(testX)

    #reshape predicted datasets
    train_pred_reshaped = np.array(train_pred).reshape(-1,1)
    test_pred_reshaped = np.array(test_pred).reshape(-1,1)
    
    #Inverse tranformed the datasets
    trainPredict = scaler.inverse_transform(train_pred_reshaped)
    testPredict = scaler.inverse_transform(test_pred_reshaped)
    trainY_inverse = scaler.inverse_transform([trainY])
    testY_inverse = scaler.inverse_transform([testY])

    #find the R2, MSE, RMSE and MAE
    R2train = r2_score(trainY_inverse[0], trainPredict[:,0])
    print('Train score: %.2f R_squared' % (R2train), file=open(pathReport, 'a'))
    R2test = r2_score(testY_inverse[0], testPredict[:,0])
    print('Test score: %.2f R_squared' % (R2test), file=open(pathReport, 'a'))

    maetrain = mean_absolute_error(trainY_inverse[0], trainPredict[:,0])
    print('Train score: %.2f MAE' % (maetrain), file=open(pathReport, 'a'))
    maetest = mean_absolute_error(testY_inverse[0], testPredict[:,0])
    print('Test score: %.2f MAE' % (maetest), file=open(pathReport, 'a'))

    msetrain = mean_squared_error(trainY_inverse[0], trainPredict[:,0])
    print('Train score: %.2f MSE' % (msetrain), file=open(pathReport, 'a'))
    msetest = mean_squared_error(testY_inverse[0], testPredict[:,0])
    print('Test score: %.2f MSE' % (msetest), file=open(pathReport, 'a'))

    trainscore = math.sqrt(mean_squared_error(trainY_inverse[0], trainPredict[:,0]))
    print('Train score: %.2f RMSE' % (trainscore), file=open(pathReport, 'a'))
    testscore = math.sqrt(mean_squared_error(testY_inverse[0], testPredict[:,0]))
    print('Test score: %.2f RMSE' % (testscore), file=open(pathReport, 'a'))

    return [trainPredict, testPredict]
    #return 0 #dummy output - zero error


#-----------------------------------------
#Ploting
#Input: ...
#Output: ...
def plottingRegression(dataset, upscalled_data, trainPredict, testPredict, seq_size, scaler, pathPlot):
    
    #Shift train prediction for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[seq_size:len(trainPredict)+seq_size, :] = trainPredict

    #shift test prediction for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(seq_size*2)+1: len(dataset)-1, :] = testPredict

    #This section is to properly visualize the floc length evolution
    train_Predicted = []

    for i in trainPredictPlot:
        train_Predicted.append(i[0])

    #perform the same operation for test_predict dataset
    test_Predicted = []

    for i in testPredictPlot:
        test_Predicted.append(i[0])

    #Create new pandas dataframe
    df1_predicted = upscalled_data[0:][['Tf']]

    df1_predicted['Predicted_train'] = train_Predicted
    df1_predicted['Predicted_test'] = test_Predicted

    #Let us tranform the original dataset for our plot.
    actual_data =scaler.inverse_transform(dataset)
    df1_predicted['Observed_data'] = actual_data

    df1_predicted.set_index ('Tf', inplace=True)

    #plot baseline and prediction
    plt.figure(figsize=(10,4), dpi=(300))
    plt.plot(df1_predicted.Observed_data, 'b', label = 'Observed')
    plt.plot(df1_predicted.Predicted_train,  'r', label = 'Predicted train')
    plt.plot(df1_predicted.Predicted_test, 'g', label = 'Predicted test')
    #plt.title('Flocs size evolution model (Gf= 60sec-1)')
    plt.title('Flocs size evolution model (Gf/sec-1)')
    plt.xlabel('Flocculation Time (minutes)')
    plt.xticks(np.arange(0,180,10))
    plt.tick_params(axis='both', which='both', color='grey')
    plt.tick_params(axis='x', which='minor', color='lightgrey') #bottom='False'
    plt.ylabel('Number of flocs')
    plt.legend(loc='upper right', fontsize=10, shadow=False)

    #Save the Chart of Floc size evolution
    plt.savefig(pathPlot)
    plt.close()

    return df1_predicted


