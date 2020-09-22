"""
    Author: Kaike Alves
    Updated at: October 20, 2018
    Python version: 3.6
    Package dependencies:
        * numpy
        * matplotlib
        * scikit-learn
    OS Dependencies (run under Linux):
        * python3.6
        * python3.6-dev
        * python3.6-venv
    
    Installation (using PIP)
    In order to run the script, follow the steps:
        $ /usr/bin/python3.6 -m venv venv --prompt="<fcast> "
        $ source venv/bin/activate ; pip install --upgrade pip setuptools
         <fcast> $ pip install numpy matplotlib scikit-learn
         <fcast> $ python michel.py
"""

import numpy as np
import matplotlib.pyplot as plt 

from model import ePLKRLS
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Biblioteca HDF5
import h5py
# Test
import pytest
# gprof2dot
import gprof2dot
# profiling tools
import pprofile

"""
Experiments with ePL-KRLS from paper
Participatory Learning in Power Transformers Thermal Modeling (Michel)
"""

# Load input/output data
X, y = np.loadtxt('X.txt'), np.loadtxt('y.txt')
# Capturing maximums for X and y
ym = y.max()
# Normalizing data from X
X, y = X / X.max(axis=0), y / y.max()

with h5py.File('data.h5','w') as hdf:
    g1 = hdf.create_group('01. Entradas')
    g1.create_dataset('X', data = X)
    g1.create_dataset('y', data = y)
    
with h5py.File('data.h5','r') as hdf:
    g1 = hdf.get('01. Entradas')
    InputX = np.array(g1.get('X'))
    Inputy = np.array(g1.get('y'))

# # Capturing maximums for X and y
ym = Inputy.max()
# # Normalizing data from X
X, y = InputX / InputX.max(axis=0), Inputy / Inputy.max()

# Creating model instance
eplkrls = ePLKRLS()
# Creating predictions and rules result lists
predictions, rules = list(), list()
# Iterating over input vector

for k in range(1,len(X)-500):
    # Capturing prediction for k + 1
    prediction = eplkrls.evolve(X[k], y[k])
    # Adding prediction to list
    predictions.append(prediction)
    # Adding rules to list
    rules.append(eplkrls.rules[-1] if len(eplkrls.rules) > 0 else 1)
for k in range(len(X)-500, len(X)):
    # Capturing prediction for k + 1
    prediction = eplkrls.evolve(X[k])
    # Adding prediction to list
    predictions.append(prediction)
    # Adding rules to list
    rules.append(eplkrls.rules[-1] if len(eplkrls.rules) > 0 else 1)
# Denormalizing data
predictions = [p * ym for p in predictions]
y = [p * ym for p in Inputy]

# Armazenando os resultados
with h5py.File('data.h5','w') as hdf:
    g2 = hdf.create_group('02. Resultados')
    g2.create_dataset('Predictions', data = predictions)
    g2.create_dataset('Rules', data = rules)

# Printing RMSE
print("RMSE = ", np.sqrt(mean_squared_error(predictions[len(X)-500:len(X)-2], y[len(X)-500:len(X)-2])))
print("NDEI = ", np.sqrt(mean_squared_error(predictions[len(X)-500:len(X)-2], y[len(X)-500:len(X)-2]))/np.std(y[len(X)-500:len(X)-2]))
print("MAE = ", mean_absolute_error(predictions[len(X)-500:len(X)-2], y[len(X)-500:len(X)-2]))

# Printing final number of rules
print("Rules = ", rules[-1])

with h5py.File('data.h5','r') as hdf:
    g2 = hdf.get('02. Resultados')
    pred = np.array(g2.get('Predictions'))
    ru = np.array(g2.get('Rules'))

# Plotting actual time series and its prediction
plt.plot(y[len(X)-500:len(X)-2], label='Actual Value', color='blue')
plt.plot(pred[len(X)-500:len(X)-2], color='red', label='ePL-KRLS')
plt.ylabel('Values')
plt.xlabel('Samples')
plt.legend()
plt.show()


# Plotting rule evolution
plt.plot(ru, color='blue')
plt.ylabel('Number of Fuzzy Rules')
plt.xlabel('Samples')
plt.show()