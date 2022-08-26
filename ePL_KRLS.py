# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:27:20 2021

@author: Kaike Sa Teles Rocha Alves
@email: kaike.alves@engenharia.ufjf.br
"""
# Importing libraries
import pandas as pd
import numpy as np
import math

class ePL_KRLS:
    def __init__(self, alpha = 0.001, beta = 0.05, lambda1 = 0.0000001, sigma = 0.5, nu = 0.25, tau = 0.05, e_utility = 0.05, d_max = 20):
        self.hyperparameters = pd.DataFrame({'alpha':[alpha],'beta':[beta], 'lambda1':[lambda1], 'sigma':[sigma], 'nu':[nu], 'tau':[tau], 'e_utility':[e_utility], 'd_max':[d_max]})
        self.parameters = pd.DataFrame(columns = ['Center', 'Dictionary', 'nu', 'P', 'K_inv', 'Theta','ArousalIndex', 'Utility', 'SumLambda', 'TimeCreation', 'CompatibilityMeasure', 'OldCenter', 'tau', 'lambda', 'Old'])
        # Evolution of the model rules
        self.rules = []
        # Computing the output in the training phase
        self.OutputTrainingPhase = np.array([])
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = np.array([])
        # Computing the output in the testing phase
        self.OutputTestPhase = np.array([])
        # Computing the residual square in the testing phase
        self.ResidualTestPhase = np.array([])
        # Verify if a new rule was created
        self.NewRule = 0
         
    def fit(self, X, y):
        # Prepare the first input vector
        x = X[0,].reshape((1,-1)).T
        # Initialize the first rule
        self.Initialize_First_Cluster(x, y[0])
        for k in range(1, X.shape[0]):
            # print(k)
            # if k == 7439:
            #     print(k)
            self.NewRule = 0
            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
            # Compute the compatibility measure and the arousal index for all rules
            for i in self.parameters.index:
                self.Compatibility_Measure(x, i)
                self.Arousal_Index(i)
            # Find the minimum arousal index
            MinIndexArousal = self.parameters['ArousalIndex'].astype('float64').idxmin()
            # Find the maximum compatibility measure
            MaxIndexCompatibility = self.parameters['CompatibilityMeasure'].astype('float64').idxmax()
            # Verifying the needing to creating a new rule
            if self.parameters.loc[MinIndexArousal, 'ArousalIndex'] > self.hyperparameters.loc[0, 'tau']:
                self.Initialize_Cluster(x, y[k], k+1, MaxIndexCompatibility)
                self.NewRule = 1
            else:
                self.Rule_Update(x, y[k], MaxIndexCompatibility)
            self.Lambda(x)
            if self.parameters.shape[0] > 1:
                self.Utility_Measure(X[k,], k+1)
            self.rules.append(self.parameters.shape[0])
            # # Finding the maximum compatibility measure
            # MaxIndexCompatibility = self.parameters['CompatibilityMeasure'].astype('float64').idxmax()
            # if self.NewRule == 0:
            #     self.KRLS(x, y[k], MaxIndexCompatibility, k+1)    
            if self.NewRule == 0:
                for row in self.parameters.index:
                    self.KRLS(x, y[k], row, k+1)
            else:
                for row in range(self.parameters.shape[0] - 1):
                    self.KRLS(x, y[k], self.parameters.index[row], k+1)
            # Computing the output
            Output = 0
            for row in self.parameters.index:
                yi = 0
                for ni in range(self.parameters.loc[row, 'Dictionary'].shape[1]):
                    yi = yi + self.parameters.loc[row, 'Theta'][ni] * self.Kernel_Gaussiano(self.parameters.loc[row, 'Dictionary'][:,ni].reshape(-1,1), x)
                Output = Output + yi * self.parameters.loc[row, 'lambda']
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y[k])**2)
        return self.OutputTrainingPhase, self.rules
            
    def predict(self, X):
        for k in range(X.shape[0]):
            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
            # Computing the compatibility measure
            for i in self.parameters.index:
                self.Compatibility_Measure(x, i)
            self.Lambda(x)
            # Computing the output
            Output = 0
            for row in self.parameters.index:
                yi = 0
                for ni in range(self.parameters.loc[row, 'Dictionary'].shape[1]):
                    yi = yi + self.parameters.loc[row, 'Theta'][ni] * self.Kernel_Gaussiano(self.parameters.loc[row, 'Dictionary'][:,ni].reshape(-1,1), x)
                Output = Output + yi * self.parameters.loc[row, 'lambda']
            self.OutputTestPhase = np.append(self.OutputTestPhase, Output)
        return self.OutputTestPhase
        
    def Initialize_First_Cluster(self, x, y):
        Q = np.linalg.inv(np.ones((1,1)) * (self.hyperparameters.loc[0, 'lambda1'] + (self.Kernel_Gaussiano(x, x))))
        Theta = Q*y
        self.parameters = pd.DataFrame([[x, x, self.hyperparameters.loc[0, 'nu'], np.ones((1,1)), Q, Theta, 0., 1., 0., 1., 1., 1., np.zeros((x.shape[0],1)), 1., np.ones((1,1))]], columns = ['Center', 'Dictionary', 'nu', 'P', 'K_inv', 'Theta', 'ArousalIndex', 'Utility', 'SumLambda', 'NumObservations', 'TimeCreation', 'CompatibilityMeasure', 'OldCenter', 'tau', 'Old'])
        Output = y
        self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y)**2)
    
    def Initialize_Cluster(self, x, y, k, i):
        Q = np.linalg.inv(np.ones((1,1)) * (self.hyperparameters.loc[0, 'lambda1'] + (self.Kernel_Gaussiano(x, x))))
        Theta = Q*y
        #nu = (np.linalg.norm(x - self.parameters.loc[i, 'Center'])/math.sqrt(-2 * np.log(max(self.epsilon))))
        NewRow = pd.DataFrame([[x, x, self.hyperparameters.loc[0, 'nu'], np.ones((1,1)), Q, Theta, 0., 1., 0., 1., k, 1., np.zeros((x.shape[0],1)), 1., np.ones((1,1)) * k]], columns = ['Center', 'Dictionary', 'nu', 'P', 'K_inv', 'Theta', 'ArousalIndex', 'Utility', 'SumLambda', 'NumObservations', 'TimeCreation', 'CompatibilityMeasure', 'OldCenter', 'tau', 'Old'])
        self.parameters = pd.concat([self.parameters, NewRow], ignore_index=True)
    
    def Kernel_Gaussiano(self, Vector1, Vector2):
        return math.exp(-((((np.linalg.norm(Vector1-Vector2))**2)/(2*self.hyperparameters.loc[0, 'sigma']**2))))
    
    def Compatibility_Measure(self, x, i):
        self.parameters.at[i, 'CompatibilityMeasure'] = (1 - ((np.linalg.norm(x - self.parameters.loc[i, 'Center']))/x.shape[0]))
            
    def Arousal_Index(self, i):
        self.parameters.at[i, 'ArousalIndex'] = self.parameters.loc[i, 'ArousalIndex'] + self.hyperparameters.loc[0, 'beta'] * (1 - self.parameters.loc[i, 'CompatibilityMeasure'] - self.parameters.loc[i, 'ArousalIndex'])
    
    def Rule_Update(self, x, y, i):
        # Update the number of observations in the rule
        self.parameters.at[i, 'NumObservations'] = self.parameters.loc[i, 'NumObservations'] + 1
        # Store the old cluster center
        self.parameters.at[i, 'OldCenter'] = self.parameters.loc[i, 'Center']
        # Update the cluster center
        self.parameters.at[i, 'Center'] = self.parameters.loc[i, 'Center'] + (self.hyperparameters.loc[0, 'alpha'] * (self.parameters.loc[i, 'CompatibilityMeasure'])**(1 - self.parameters.loc[i, 'ArousalIndex'])) * (x - self.parameters.loc[i, 'Center'])
            
    def mu(self, x1, x2, row, j):
        mu = math.exp( - ( x1 - x2 ) ** 2 / ( 2 * self.parameters.loc[row, 'sigma'][j,0] ** 2 ) )
        return mu
    
    def tau(self, x):
        for row in self.parameters.index:
            tau = 1
            for j in range(x.shape[0]):
                tau = tau * self.mu(x[j], self.parameters.loc[row, 'Center'][j,0], row, j)
            # Evoid mtau with values zero
            if abs(tau) < (10 ** -100):
                tau = (10 ** -100)
            self.parameters.at[row, 'tau'] = tau
            
    def Lambda(self, x):
        for row in self.parameters.index:
            self.parameters.at[row, 'lambda'] = self.parameters.loc[row, 'tau'] / sum(self.parameters['tau'])
            self.parameters.at[row, 'SumLambda'] = self.parameters.loc[row, 'SumLambda'] + self.parameters.loc[row, 'lambda']
            
    def Utility_Measure(self, x, k):
        # Calculating the utility
        remove = []
        for i in self.parameters.index:
            if (k - self.parameters.loc[i, 'TimeCreation']) == 0:
                self.parameters.at[i, 'Utility'] = 1
            else:
                self.parameters.at[i, 'Utility'] = self.parameters.loc[i, 'SumLambda'] / (k - self.parameters.loc[i, 'TimeCreation'])
            if self.parameters.loc[i, 'Utility'] < self.hyperparameters.loc[0, 'e_utility']:
                remove.append(i)
        if len(remove) > 0:    
            self.parameters = self.parameters.drop(remove)
            
    def KRLS(self, x, y, i, k):
        # Update the kernel size
        #self.parameters.at[i, 'nu'] = math.sqrt((self.parameters.loc[i, 'nu'])**2 + (((np.linalg.norm(x - self.parameters.loc[i, 'Center']))**2 - (self.parameters.loc[i, 'nu'])**2)/self.parameters.loc[i, 'NumObservations']) + ((self.parameters.loc[i, 'NumObservations'] - 1) * ((np.linalg.norm(self.parameters.loc[i, 'Center'] - self.parameters.loc[i, 'OldCenter']))**2))/self.parameters.loc[i, 'NumObservations'])
        # Compute k
        kt = np.array(())
        for ni in range(self.parameters.loc[i, 'Dictionary'].shape[1]):
            kt = np.append(kt, [self.Kernel_Gaussiano(self.parameters.loc[i, 'Dictionary'][:,ni].reshape(-1,1), x)])
        kt = kt.reshape(-1,1)
        # Computing a
        a = np.matmul(self.parameters.loc[i, 'K_inv'], kt).reshape(-1,1)
        #A = a.reshape(a.shape[0],1)
        # Computing delta
        delta = self.hyperparameters.loc[0, 'lambda1'] + 1 - np.matmul(kt.T, a)
        # Searching for the lowest distance between the input and the dictionary inputs
        distance = []
        for ni in range(self.parameters.loc[i, 'Dictionary'].shape[1]):
            distance.append(np.linalg.norm(self.parameters.loc[i, 'Dictionary'][:,ni].reshape(-1,1) - x))
        # Finding the index of minimum distance
        IndexMinDistance = np.argmin(distance)
        # Estimating the error
        EstimatedError = (y - np.matmul(kt.T, self.parameters.loc[i, 'Theta']) )
        # Novelty criterion
        if distance[IndexMinDistance] > 0.1 * self.parameters.loc[i, 'nu'] and self.parameters.loc[i,'Dictionary'].shape[1] < self.hyperparameters.loc[0, 'd_max']:
            self.parameters.at[i, 'Dictionary'] = np.hstack([self.parameters.loc[i, 'Dictionary'], x])
            self.parameters.at[i, 'Old'] = np.concatenate([self.parameters.loc[i, 'Old'], [[k]]], axis=1)
            # Updating Q                      
            self.parameters.at[i, 'K_inv'] = (1/delta)*(self.parameters.loc[i, 'K_inv']*delta + np.matmul(a,a.T))
            self.parameters.at[i, 'K_inv'] = np.lib.pad(self.parameters.loc[i, 'K_inv'], ((0,1),(0,1)), 'constant', constant_values=(0))
            sizeQ = self.parameters.loc[i, 'K_inv'].shape[0] - 1
            self.parameters.at[i, 'K_inv'][sizeQ,sizeQ] = (1/delta) #* self.hyperparameters.loc[0, 'omega']
            self.parameters.at[i, 'K_inv'][0:sizeQ,sizeQ] = (1/delta)*(-a.flatten())
            self.parameters.at[i, 'K_inv'][sizeQ,0:sizeQ] = (1/delta)*(-a.flatten())
            # Updating P
            self.parameters.at[i, 'P'] = np.lib.pad(self.parameters.loc[i, 'P'], ((0,1),(0,1)), 'constant', constant_values=(0))
            sizeP = self.parameters.loc[i, 'P'].shape[0] - 1
            self.parameters.at[i, 'P'][sizeP,sizeP] = 1
            # Updating Theta
            self.parameters.at[i, 'Theta'] = self.parameters.loc[i, 'Theta'] - ( (a/delta) * EstimatedError )
            self.parameters.at[i, 'Theta'] = np.vstack([self.parameters.loc[i, 'Theta'], (1/delta) * EstimatedError])
        else:
            # d_old = np.argmin(self.parameters.loc[i, 'Old'], axis=1)
            # self.parameters.at[i, 'Old'][0, d_old] = k
            # self.parameters.at[i, 'Dictionary'][:,d_old] = x
            # Calculating q
            q = np.matmul(self.parameters.loc[i, 'P'], a)/(1 + np.matmul(np.matmul(a.T, self.parameters.loc[i, 'P']), a))
            # Updating P
            self.parameters.at[i, 'P'] = self.parameters.loc[i, 'P'] - (np.matmul(np.matmul(np.matmul(self.parameters.loc[i, 'P'],a), a.T), self.parameters.loc[i, 'P']))/(1 + np.matmul(np.matmul(a.T, self.parameters.loc[i, 'P']), a))
            # Updating Theta
            self.parameters.at[i, 'Theta'] = self.parameters.loc[i, 'Theta'] + np.matmul(self.parameters.loc[i, 'K_inv'], q) * EstimatedError