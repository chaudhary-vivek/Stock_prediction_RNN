# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

# The column "Open" are stored in training_set
training_set = dataset_train.iloc[:, 1:2].values

# Feature scaling
# All values are sacled down in the range 0,1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating datastructue with 60 timesteps and one output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# RNN requires the input to be of the dimension (60,1,1)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Building rnn

# Importing required libraries
# Dropout drops the neurons with zero as output
# Dense layer is for the output
# LSTM is for other neurons
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout 

# Initialising RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
# Dropout drops 20% of the outputs
# input shape is one timestamp
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
# This is the output layer with just one output
regressor.add(Dense(units = 1))

# Compiling the RNN
# Optimizer used is adam and the loss measured is the mean squared error
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
 
# Fitting the RNN to the Training set
# Input is in batch sizes of 30
regressor.fit(X_train, y_train, epochs = 50, batch_size = 32)

# Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
# First, append data from past 60 months to test data
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

# Reshaping the test data in the required dimensions (60,1,1)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Making prediction using RNN
predicted_stock_price = regressor.predict(X_test)

# Reverse transforming scaled data to normal data
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results on two graphs
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

# Red graph is the real value
# Blue graph is the predicted value
# The predicted value follows the real value very closely
# Predicted value improves with the number of epochs
