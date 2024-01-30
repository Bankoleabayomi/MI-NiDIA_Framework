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


#------------------------
#Data reading and preprocessing (upscaling and formatting)
#Input: ...
#Output: ...
def preprocessing(pathData,upscalOpt):
    
    #Reading input data
    df = pd.read_csv(pathData)

    #Preprocessing stage: Upscaling and formatting.
    df.index.rename('SN', inplace = True)
    df['Time'] = pd.to_datetime(df['Time'])
    Gf_upscaled = df.resample('S', on = 'Time').mean()
    upscalled_data = Gf_upscaled.interpolate(method='linear')

    #select your choice floc size range here incase there are multiple groups in the file
    choice_group = upscalled_data[upscalOpt]
    choice_group.head(10)

    #convert data to numpy array
    dataset = choice_group.values
    dataset = dataset.astype('float32')

    #nomalize the dataset because some activation function are sensitive to high magnitude of data values
    scaler = MinMaxScaler(feature_range=(0,1))
    dataset = scaler.fit_transform(np.array(dataset).reshape(-1,1))

    return [dataset,scaler,upscalled_data]



#------------------------
#Dataset organized as a sequence
#Input: ...
#Output: ...
def to_sequences(dataset, seq_size =1):
  x= []
  y= []
  for i in range (len(dataset)-seq_size-1):
    #print i
    window = dataset[i:(i+seq_size), 0]
    x.append(window)
    y.append(dataset[(i+seq_size), 0])

  return np.array(x), np.array(y)



#------------------------
#Dataset organization for training and validation procedures
#Input: ...
#Output: ...
def dataTVS(dataset,trainingPercentage,seq_size, optModel):

    train_size = int(len(dataset)*trainingPercentage)
    test_size = len(dataset)- train_size
    train, test = dataset[0:train_size,:], dataset[train_size: len(dataset),:]

    trainX, trainY = to_sequences(train, seq_size)
    testX, testY = to_sequences(test, seq_size)

    if optModel == 'lstm':
      #Reshape the dataset into time-series vector data
      trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
      testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    if optModel == 'mlp':
      trainX, trainY = to_sequences(train, seq_size)
      testX, testY = to_sequences(test, seq_size)

    return [trainX, trainY, testX, testY]