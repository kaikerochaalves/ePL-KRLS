# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:55:26 2021

@author: Kaike Sa Teles Rocha Alves
@email: kaike.alves@engenharia.ufjf.br
"""
# Importing libraries
import math
import numpy as np
import pandas as pd
import statistics as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt 

# Defining the atributes
from statsmodels.tsa.tsatools import lagmat

# Importing the model
from ePL_KRLS import ePL_KRLS

#-----------------------------------------------------------------------------
# Importing the Power Trasnformers Dataset Day 3
#-----------------------------------------------------------------------------

# Importing the data
Data = pd.read_excel (r'Datasets\PowerTransformerDay3.xlsx', header=None)

# Total number of points
n = Data.shape[0]
# Defining the number of training points
training_size = n


# Changing to matrix
Data = Data.to_numpy()
# Separating the inputs and output
X = Data[:-1,:-1]
y = Data[:-1,-1]

def Normalize_Data(data):
    Max_data = np.max(data)
    Min_data = np.min(data)
    Normalized_data = (data - Min_data)/(Max_data - Min_data)
    return Normalized_data, Max_data, Min_data

# Spliting the data into train and test
X_train, X_test = X[:training_size,:], X[training_size:,:]
y_train, y_test = y[:training_size], y[training_size:]

# Normalize the inputs and the output
Normalized_X, X_max, X_min = Normalize_Data(X)
Normalized_y, y_max, y_min = Normalize_Data(y)

# Spliting normilized data into train and test
Normalized_X_train, Normalized_X_test = Normalized_X[:training_size,:], Normalized_X[training_size:,:]
Normalized_y_train, Normalized_y_test = Normalized_y[:training_size], Normalized_y[training_size:]

#-----------------------------------------------------------------------------
# Calling the ePL-KRLS
#-----------------------------------------------------------------------------

# Setting the hyperparameters
alpha = 0.5
tau = 0.06
beta = 0.4
lambda1 = 0.0000001
sigma = 0.5
nu = 0.4
e_utility = 0.03
d_max = 100

# Initializing the model
model = ePL_KRLS(alpha = alpha, beta = beta, lambda1 = lambda1, sigma = sigma, nu = nu, tau = tau, e_utility = e_utility, d_max = d_max)
# Train the model
OutputTraining, Rules = model.fit(Normalized_X_train, Normalized_y_train)

#-----------------------------------------------------------------------------
# Evaluate the model's performance
#-----------------------------------------------------------------------------

# Calculating the error metrics
# DeNormalize the results
OutputTrainingDenormalized = OutputTraining * (y_max - y_min) + y_min
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_train, OutputTrainingDenormalized))
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(y_train.flatten())
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_train, OutputTrainingDenormalized)

# Printing the RMSE
print("RMSE = ", RMSE)
# Printing the NDEI
print("NDEI = ", NDEI)
# Printing the MAE
print("MAE = ", MAE)
# Printing the number of final rules
print("Final Rules = ", Rules[-1])

#-----------------------------------------------------------------------------
# Plot the graphics
#-----------------------------------------------------------------------------

# Plot the graphic of the actual time series and the ePL-KRLS predictions
plt.plot(y_train, label='Actual Value', color='red')
plt.plot(OutputTrainingDenormalized, color='blue', label='ePL-KRLS')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend()
plt.show()

# Plot the evolution of the model's rule
plt.plot(Rules, color='blue')
plt.ylabel('Number of Fuzzy Rules')
plt.xlabel('Samples')
plt.show()