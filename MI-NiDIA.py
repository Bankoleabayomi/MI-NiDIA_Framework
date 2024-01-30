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

#Local modules for MI-NiDIA
import minidiaProcs
import minidiaModels
#-------------------------------------------------------------



#-----------------------------------------
#This is to downgrade the Tf to lower version for better compatibility with the libraries used in this script.
#!pip install tensorflow==2.12.0
#!pip install keras==2.12


#This code is only for the first installation of scikeras.
#try:
#    import scikeras
#except ImportError:
#    !python -m pip install scikeras



#Include a brief description about how the input data is formatted.
#...


#-----------------------------------------
#User parameters (include comments as a documentation for each parameter)
pathData = 'C:\\Users\\banko\\Downloads\Gf20_sort.csv' 
upscalOpt = 'GRP_1'
modelOpt = 'lstm' #'lstm' #or 'mlp'
trainingPercentage = 0.7
seq_size = 5 #number of time-steps
cvFolds = 3 #nummber of cross-validation folds
# pathReport = './results/LSTM_Report.txt'
# pathPlot = './results/LSTM_Floc evolution plot.png'
# pathEstimates = './results/LSTM_Estimates.cvs'
pathReport = 'C:\\Users\\banko\Downloads\Result\LSTM_Report.txt'
pathPlot = 'C:\\Users\\banko\\Downloads\Result\LSTM_Floc evolution plot.png'
pathEstimates = 'C:\\Users\\banko\\Downloads\Result\LSTM_Estimates.csv'


#-----------------------------------------
#Data reading and preprocessing (upscaling and formatting)
[dataset,scaler,upscalledData] = minidiaProcs.preprocessing(pathData,upscalOpt)


#-----------------------------------------
#Dataset organization for training and validation procedures
#[trainX, trainY, testX, testY] = minidiaProcs.dataTVS(dataset,trainingPercentage,seq_size)
[trainX, trainY, testX, testY] = minidiaProcs.dataTVS(dataset,trainingPercentage,seq_size,modelOpt)


#-----------------------------------------
#Building and tune the neural network-based model (...)
best_model = minidiaModels.buildModel(trainX, trainY, cvFolds, modelOpt, seq_size)


#-----------------------------------------
#Prediction summary
[trainPredict, testPredict] = minidiaModels.modelSummary(trainX, trainY, testX, testY, scaler, best_model, pathReport, modelOpt)


#-----------------------------------------
#Ploting output and saving the model outputs
plotDF = minidiaModels.plottingRegression(dataset, upscalledData, trainPredict, testPredict, seq_size, scaler, pathPlot)
plotDF.to_csv(pathEstimates,sep=';')


print('End of process. Check saved results at: \n%s \n%s \n%s '  % (pathReport,pathPlot,pathEstimates))
