# -*- coding: utf-8 -*-
"""
Script to load covid data and make predictions and store them

@author: mariano
"""

#-------------------------------
# Import libraries
#-------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import numpy
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import settings
import sqlite3


#-------------------------------
# Load data 
#-------------------------------

# Get user name defined in settings
User_Name = settings.User_Name

# Load covid data 
df = pd.read_csv("C:/Users/{}/Covid/Data/Covid_Country.csv".format(User_Name))


#----------------------------------------
# Plot a country framework
#----------------------------------------

df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')

x = df['date']
y = df['rolling']

fig, ax = plt.subplots()
ax.plot_date(x, y, markerfacecolor='CornflowerBlue', markeredgecolor='white')
fig.autofmt_xdate()


#---------------------------
# Prepare data for training
#---------------------------

# Fix random seed for reproducibility
numpy.random.seed(7)

# Use the dataframe from here to build machine learning model
data = df.copy()

# Use the rolling column as timeseries
dataset = data[['rolling']]
dataset = dataset.values

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Split into train and test sets
train_size = int(len(dataset) * 0.90)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# Convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

# Reshape into X=t and Y=t+1
look_back = 7
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))


#---------------------------
# Train model 
#---------------------------

# Create and fit the LSTM network
model = Sequential()
model.add(LSTM(3, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=5, verbose=0)

# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


#---------------------------
# Calculate error
#---------------------------

# Calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


#---------------------------
# Plot graph
#---------------------------

# Shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# Shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2):len(dataset), :] = testPredict

# Plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)


plt.show()


#--------------------------------------
# Extract last value of the testing 
#--------------------------------------

# Show the last element predicted  
testPredict_length = len(testPredict)
Pred_last_element = testPredict[testPredict_length - 1]
print(Pred_last_element)

# Show the last real element  testY
Flat_TestY = testY.flatten()
testY_length = len(Flat_TestY)
Real_last_element = Flat_TestY[testY_length-1]
print(Real_last_element)

# Prepare prediction for plotting
Predict_Point_Plot = numpy.empty_like(dataset)
Predict_Point_Plot[:, :] = numpy.nan
Predict_Point_Plot[len(Predict_Point_Plot)-1] = Pred_last_element

# Prepare last real point for plotting
Real_Point_Plot = numpy.empty_like(dataset)
Real_Point_Plot[:, :] = numpy.nan
Real_Point_Plot[len(Real_Point_Plot)-1] = Real_last_element

# Plot it
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.plot(Predict_Point_Plot,'go',marker=".",markersize=20)
plt.plot(Real_Point_Plot,'bo',marker=".",markersize=20)

plt.show()


#-------------------------------------------------
# Pred new data sample
#-------------------------------------------------

# Use original data 
data_pred = df.copy()

# Get the last elements to be used to make new prediction
data_pred_os = data_pred[-look_back:]

# Use the rolling column as timeseries
dataset_os = data_pred_os[['rolling']]
dataset_os = dataset_os.values

# Normalize the dataset
dataset_os = scaler.transform(dataset_os)

# Reshape input to be [samples, time steps, features]
dataset_os = numpy.reshape(dataset_os, (dataset_os.shape[1], dataset_os.shape[0], 1))

# Make predictions
dt_pred = model.predict(dataset_os)

# Invert predictions
dt_pred = scaler.inverse_transform(dt_pred)


#----------------------------------------------------------
# Save last prediction into a file for further anlysis
#----------------------------------------------------------

### Save needed data ###

# Get date of last current value
Current_Date = df.loc[len(df)-1,'date']

# Tomorrow date 
Tomorrow_Date = Current_Date + timedelta(days=1)

# Pass data to dataframe 
data = {'Real_Point'      : [int(Real_last_element)],
        'Predicted_Point' : [int(Pred_last_element[0])],
        'Pred_Tomorrow'   : [int(dt_pred[0,0])],
        'Date'            : Current_Date,
        'Tomorrow_Date'   :Tomorrow_Date}

df_pred = pd.DataFrame(data, columns = ['Date','Real_Point', 
                                        'Predicted_Point', 
                                        'Tomorrow_Date', 
                                        'Pred_Tomorrow'])

### Save as cvs ###

df_pred.to_csv("C:/Users/{}/Covid/Data/Pred_Covid.csv".format(User_Name))


### Save as database ###

# Create a sql database
conn = sqlite3.connect('C:/Users/{}/Covid/Data/Predictions_Covid.db'.format(User_Name))
c = conn.cursor()

# Create a table inside the sql database Coins from the dataframe 
df_pred.to_sql('RESULTS', conn, if_exists='replace', index = False)

# Commit the changes of the database
conn.commit()

# Close the connection to the database
conn.close()



































