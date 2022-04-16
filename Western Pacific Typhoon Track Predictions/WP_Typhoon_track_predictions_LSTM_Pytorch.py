#!/usr/bin/env python
# coding: utf-8

import pandas as pd 
from selenium import webdriver
from selenium.webdriver.common.by import By
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from bs4 import BeautifulSoup
import requests
import re
import string


full_WP = pd.read_csv("./Full WP dataset.csv")


# ### Data Preprcoessing


full_WP = full_WP.iloc[1:]



full_WP = full_WP.loc[full_WP['SEASON'].astype(int) >= 2000]


columns_features = ['ISO_TIME', 'SEASON', 'NAME', 'USA_LAT', 'USA_LON', 'DIST2LAND', 'USA_WIND', 'USA_PRES', 'USA_R34_NE', 'USA_R34_SE', 'USA_R34_SW', 'USA_R34_NW', 'USA_R50_NE', 'USA_R50_SE', 'USA_R50_SW', 'USA_R50_NW', 'USA_R64_NE', 'USA_R64_SE', 'USA_R64_SW', 'USA_R64_NW']


full_WP_modify = full_WP[columns_features].copy()


full_WP_modify = full_WP_modify.reset_index(drop=True)


full_WP_modify = full_WP_modify[full_WP_modify.NAME != 'NOT_NAMED']


full_WP_modify = full_WP_modify.reset_index(drop=True)


full_WP_modify = full_WP_modify[full_WP_modify.USA_LAT!=' ']


full_WP_modify = full_WP_modify.replace(r'^\s*$', -1 , regex=True)


from sklearn.preprocessing import MinMaxScaler

scaler_lat = MinMaxScaler()
scaler_long = MinMaxScaler()
scaler_dist = MinMaxScaler()
scaler_wind = MinMaxScaler()
scaler_pres = MinMaxScaler()
scaler_r34_ne = MinMaxScaler()
scaler_r34_se = MinMaxScaler()
scaler_r34_sw = MinMaxScaler()
scaler_r34_nw = MinMaxScaler()
scaler_r50_ne = MinMaxScaler()
scaler_r50_se = MinMaxScaler()
scaler_r50_sw = MinMaxScaler()
scaler_r50_nw = MinMaxScaler()
scaler_r64_ne = MinMaxScaler()
scaler_r64_se = MinMaxScaler()
scaler_r64_sw = MinMaxScaler()
scaler_r64_nw = MinMaxScaler()


full_WP_modify[['USA_LAT']] = scaler_lat.fit_transform(full_WP_modify[['USA_LAT']])
full_WP_modify[['USA_LON']] = scaler_long.fit_transform(full_WP_modify[['USA_LON']])
full_WP_modify[['DIST2LAND']] = scaler_dist.fit_transform(full_WP_modify[['DIST2LAND']])
full_WP_modify[['USA_WIND']] = scaler_wind.fit_transform(full_WP_modify[['USA_WIND']])
full_WP_modify[['USA_PRES']] = scaler_pres.fit_transform(full_WP_modify[['USA_PRES']])
full_WP_modify[['USA_R34_NE']] = scaler_r34_ne.fit_transform(full_WP_modify[['USA_R34_NE']])
full_WP_modify[['USA_R34_SE']] = scaler_r34_se.fit_transform(full_WP_modify[['USA_R34_SE']])
full_WP_modify[['USA_R34_SW']] = scaler_r34_sw.fit_transform(full_WP_modify[['USA_R34_SW']])
full_WP_modify[['USA_R34_NW']] = scaler_r34_nw.fit_transform(full_WP_modify[['USA_R34_NW']])
full_WP_modify[['USA_R50_NE']] = scaler_r50_ne.fit_transform(full_WP_modify[['USA_R50_NE']])
full_WP_modify[['USA_R50_SE']] = scaler_r50_se.fit_transform(full_WP_modify[['USA_R50_SE']])
full_WP_modify[['USA_R50_SW']] = scaler_r50_sw.fit_transform(full_WP_modify[['USA_R50_SW']])
full_WP_modify[['USA_R50_NW']] = scaler_r50_nw.fit_transform(full_WP_modify[['USA_R50_NW']])
full_WP_modify[['USA_R64_NE']] = scaler_r64_ne.fit_transform(full_WP_modify[['USA_R64_NE']])
full_WP_modify[['USA_R64_SE']] = scaler_r64_se.fit_transform(full_WP_modify[['USA_R64_SE']])
full_WP_modify[['USA_R64_SW']] = scaler_r64_sw.fit_transform(full_WP_modify[['USA_R64_SW']])
full_WP_modify[['USA_R64_NW']] = scaler_r64_nw.fit_transform(full_WP_modify[['USA_R64_NW']])

for year in range(2000,2022):
     globals()['df_WP_%s' % year] = full_WP_modify[(full_WP_modify["SEASON"]==year)]


for year in range(2000,2022):
    globals()['df_WP_%s' % year] = globals()['df_WP_%s' % year].reset_index(drop=True)


for year in range(2000,2022):
    globals()['df_WP_%s' % year] =  globals()['df_WP_%s' % year].drop(['SEASON'], axis=1)


for year in range(2000,2022):
    for i in range(len(globals()['df_WP_%s' % year])):
        globals()['df_WP_%s' % year].iloc[i,1] = re.sub(r"[^\w\s]", '',  globals()['df_WP_%s' % year].iloc[i,1])


summarylist = []
for year in range(2000,2022):
    summarylist.append(dict(tuple(globals()['df_WP_%s' % year].groupby('NAME'))))

tc_name_list = []
for i in range(len(summarylist)):
    tc_name_list.append(list(summarylist[i].keys()))


print(tc_name_list[0])


count = 0
for i in range(len(tc_name_list)):
    for name in tc_name_list[i]:
        count = count+1
        globals()['df_WP_%s_%s'%((str(2000+i)), name)] = globals()['df_WP_%s' % (str(2000+i))][globals()['df_WP_%s' % (str(2000+i))].NAME == name]
#         globals()['df_WP_%s_%s'%((str(2000+i)), name)]  = globals()['df_WP_%s_%s'%((str(2000+i)), name)][globals()['df_WP_%s_%s'%((str(2000+i)),name)].USA_LAT != ' ']
        globals()['df_WP_%s_%s'%((str(2000+i)), name)]  = globals()['df_WP_%s_%s'%((str(2000+i)), name)] .reset_index(drop=True)



for i in range(len(tc_name_list)):
    for name in tc_name_list[i]:
        globals()['df_WP_%s_%s'%((str(2000+i)), name)] = (globals()['df_WP_%s_%s'%((str(2000+i)), name)]).drop(['NAME'], axis=1)



for i in range(len(tc_name_list)):
    for name in tc_name_list[i]:
        globals()['df_WP_%s_%s'%((str(2000+i)), name)].rename(columns={'ISO_TIME':'Date'}, inplace=True)
        globals()['df_WP_%s_%s'%((str(2000+i)), name)]['Date'] = pd.to_datetime(globals()['df_WP_%s_%s'%((str(2000+i)), name)]['Date'], dayfirst=True)


cols_features = list(df_WP_2018_MANGKHUT)[1:len(df_WP_2018_MANGKHUT.columns)]
print(cols_features)


# ### Augmented Dickey Fuller test


from statsmodels.tools.sm_exceptions import (
    CollinearityWarning,
    InfeasibleTestError,
    InterpolationWarning,
    MissingDataError,
)


features = list(df_WP_2018_MANGKHUT)[1:]



features



# def Average(lst):
#     return sum(lst) / len(lst)

# def Minimum(lst):
#     return min(lst)


# from statsmodels.tsa.stattools import adfuller

# for feature in features:
#     locals()["adf"+feature] = []

# for i in range(len(tc_name_list)):
#     for name in tc_name_list[i]:
#         for feature in features:
#             # print("Stationary test of ", feature)
#             df_adf = globals()['df_WP_%s_%s'%((str(2000+i)), name)][[feature]]
#             # df_adf_lat.plot(figsize=(8,4))
#             # plt.show()
#             adf = adfuller(df_adf)
#             # print(adf[1])
#             if np.isnan(adf[1])== True:
#                 temp_list = list(adf)
#                 temp_list[1] = 0
#                 adf = tuple(temp_list)
#                 locals()["adf"+feature].append(adf[1])
#             else:
#                 locals()["adf"+feature].append(adf[1])
#             # print("ADF Stat: " , adf[0])
#             # print("P-value: " , adf[1])
#             # print("Critical Value: " , adf[4])
#             # print("\n")



# print("Threshold : 0.05 ")
# for feature in features:
#     print("Average P-value of ADF test of each", feature, "in all typhoon data :", Average(locals()["adf"+feature]))
# print("\n") 
# for feature in features:
#     print("Minimum P-value of ADF test of each", feature, "in all typhoon data :", Minimum(locals()["adf"+feature]))


# P-value is greater than significant level of 0.05, all variables are individually non stationary, therefore, pass to cointegration test to check whether residual of two variables has a long term stationary property.

# ### Cointegration test


# import statsmodels.tsa.stattools as ts
# for feature in features:
#     locals()["coin_lat_"+feature] = []
#     locals()["coin_lon_"+feature] = []

# for i in range(len(tc_name_list)):
#     for name in tc_name_list[i]:    
#         for feature in features:
#             # print("Cointegration test between USA LAT and", feature)
#             coin_result = ts.coint(globals()['df_WP_%s_%s'%((str(2000+i)), name)]["USA_LAT"], globals()['df_WP_%s_%s'%((str(2000+i)), name)][feature])
#             locals()["coin_lat_"+feature].append(coin_result[1])
#             # print("t-statistics of unit root test: ", coin_result[0])
#             # print("p-value: ", coin_result[1])
#             # print("critical values of 1%m 5% and 10%: ",coin_result[2])
#             # print("\n")

# print("----------------------------------------------------------------------------------")

# for i in range(len(tc_name_list)):
#     for name in tc_name_list[i]:    
#         for feature in features:
#             # print("Cointegration test between USA LON and", feature)
#             coin_result = ts.coint(globals()['df_WP_%s_%s'%((str(2000+i)), name)]["USA_LON"], globals()['df_WP_%s_%s'%((str(2000+i)), name)][feature])
#             locals()["coin_lon_"+feature].append(coin_result[1])
#             # print("t-statistics of unit root test: ", coin_result[0])
#             # print("p-value: ", coin_result[1])
#             # print("critical values of 1%m 5% and 10%: ",coin_result[2])
#             # print("\n")

# print("Threshold : 0.05 ")
# for feature in features:
#     print("Average P-value of Cointegration test between Latitude and", feature, "in all typhoon data :", Average(locals()["coin_lat_"+feature]))
# print("\n") 

# for feature in features:
#     print("Average P-value of Cointegration test between Longitude and", feature, "in all typhoon data :", Average(locals()["coin_lon_"+feature]))
# print("\n") 

# for feature in features:
#     print("Minimum P-value of Cointegration test between Latitude and", feature, "in all typhoon data :", Minimum(locals()["coin_lat_"+feature]))

# print("\n") 
# for feature in features:    
#     print("Minimum P-value of Cointegration test between Longitude and", feature, "in all typhoon data :", Minimum(locals()["coin_lon_"+feature]))


# All p-value is greater than 0.05 
# 
# no stationary property between latitude, longitude and other meteorological features

# Since the sequential data is not stationary, we can expect that there is no granger casuality between any features  to latitude or lonigtude

# But we can still proceed the granger causality test to clarity whether the null hypothesis of no granger casusality relationship will be accepted 

# ### Granger Causality test

# from statsmodels.tsa.stattools import grangercausalitytests


# for feature in features:
#     locals()["gc_lat_"+feature] = []

# for i in range(len(tc_name_list)):
#     for name in tc_name_list[i]:
#         for feature in features:
#             # print("Result: ", features[0], "and", feature) 
#             df_granger = globals()['df_WP_%s_%s'%((str(2000+i)), name)][[features[0], feature]]
#             # df_granger = df_WP_2018_WUKONG[[features[0], feature]]
#             try:
#                 ### 5 lags trial ### 
#                 gc_result = grangercausalitytests(df_granger, 5, verbose=False)
#                 p_value = gc_result[5][0]['ssr_ftest'][1]
#                 locals()["gc_lat_"+feature].append(p_value)
#             except InfeasibleTestError:
#                 print("The x values include a column with constant values and so the test statistic cannot be computed.")
#             except ValueError:
#                 continue
#             print("-----------------------------------------------------------------------------------")



# print("Threshold : 0.05 ")
# for feature in features:
#     print("Average P-value of Granger causality test in 5 lags between Latitude and", feature, "in all typhoon data :", Average(locals()["gc_lat_"+feature]))
# print("\n") 
# for feature in features:
#     print("Minimum P-value of Granger causality test in 5 lags between Latitude and", feature, "in all typhoon data :", Minimum(locals()["gc_lat_"+feature]))


# Based on the p-value obtained from F-test, all values are greater than the significance level of 0.05, we are failed to reject the null hypothesis. We can conclude that there is no granger causality relationship between any meteorological features and latitude, longitude

# ### Latitude Prediction


tc_name_list[18].remove('MANGKHUT')



cols_target = list(df_WP_2018_MANGKHUT)[1:2]



cols_target


print(cols_features)
print(cols_target)


for i in range(len(tc_name_list)):
    for name in tc_name_list[i]:
        globals()['df_features_lat_%s_%s'%((str(2000+i)), name)] = globals()['df_WP_%s_%s'%((str(2000+i)), name)][cols_features].astype(float)
        globals()['df_target_lat_%s_%s'%((str(2000+i)), name)] = globals()['df_WP_%s_%s'%((str(2000+i)), name)][cols_target].astype(float)

trainX_lat = []
trainY_lat = []
valX_lat = []
valY_lat = []
testX_lat = []
testY_lat = []


for i in range(len(tc_name_list)): ### 20 years of data for training (2000-2020)
    for name in tc_name_list[i]:
        globals()['df_features_lat_%s_%s'%((str(2000+i)), name)] = globals()['df_features_lat_%s_%s'%((str(2000+i)), name)].to_numpy()
        globals()['df_target_lat_%s_%s'%((str(2000+i)), name)] = globals()['df_target_lat_%s_%s'%((str(2000+i)), name)].to_numpy()


n_future = 1
n_past = 5

for i in range(len(tc_name_list)):
    for name in tc_name_list[i]:
            for j in range(n_past, len(globals()['df_features_lat_%s_%s'%((str(2000+i)), name)])-n_future+1):
                trainX_lat.append(globals()['df_features_lat_%s_%s'%((str(2000+i)), name)][j-n_past:j,0:globals()['df_features_lat_%s_%s'%((str(2000+i)), name)].shape[1]])
                trainY_lat.append(globals()['df_target_lat_%s_%s'%((str(2000+i)), name)][j+n_future-1:j+n_future,0])



trainX_lat = np.array(trainX_lat)


trainX_lat, trainY_lat= np.array(trainX_lat), np.array(trainY_lat)
valX_lat, valY_lat= np.array(valX_lat), np.array(valY_lat)
testX_lat, testY_lat= np.array(testX_lat), np.array(testY_lat)


print(trainX_lat.shape)
print(trainY_lat.shape)
print(valX_lat.shape)
print(valY_lat.shape)
print(testX_lat.shape)
print(testY_lat.shape)




from sklearn.model_selection import train_test_split

trainX_lat, valX_lat, trainY_lat, valY_lat = train_test_split(trainX_lat,trainY_lat,test_size=0.30, random_state=42)
valX_lat, testX_lat, valY_lat, testY_lat = train_test_split(valX_lat,valY_lat,test_size=0.50, random_state=42)


print(trainX_lat.shape)
print(trainY_lat.shape)
print(valX_lat.shape)
print(valY_lat.shape)
print(testX_lat.shape)
print(testY_lat.shape)


import torch

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from torch import nn, optim

import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


trainX_lat = torch.from_numpy(trainX_lat).float()
trainY_lat = torch.from_numpy(trainY_lat).float()
valX_lat = torch.from_numpy(valX_lat).float()
valY_lat = torch.from_numpy(valY_lat).float()
testX_lat = torch.from_numpy(testX_lat).float()
testY_lat = torch.from_numpy(testY_lat).float()

from torch.utils.data import TensorDataset, DataLoader
batch_size = 16
n_features = 17
n_epochs = 100

train = TensorDataset(trainX_lat, trainY_lat)
val = TensorDataset(valX_lat, valY_lat)
test = TensorDataset(testX_lat, testY_lat)

# shuffle = true
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader_one = DataLoader(test, batch_size=1, shuffle=True, drop_last=True)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out

class Optimization_Lat:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, x, y):
        # Sets model to train mode
        self.model.train()

        
        print(x)
        print(y)
        
        
        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()
    
    def train(self, train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=n_features):
        # model_path = f'models/{self.model}_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        
        model_path = "model_lat"
        validation_list = []
        previous_validation_loss = 100
        patience = 4
        count = 0
        
        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, 5, n_features])
                # print("X batch after view" , x_batch)
                # print("y batch", y_batch)
                y_batch = y_batch
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, 5, n_features])
                    y_val = y_val
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)
                print(f"[{epoch}/{n_epochs}] Training loss: {training_loss:.8f}\t Validation loss: {validation_loss:.8f}")
                validation_list.append(validation_loss)
                validation_list.sort()
                if validation_loss <= validation_list[0]:
                    torch.save(self.model, model_path)
                    print("Updated best model")
                if validation_loss > previous_validation_loss:
                    count +=1
                    print("Current Validation_loss is greater than Previous Validation Loss, counting:", count)
                    if count>=patience:
                        print("Early stopping executed!")
                        break
                else:
                    print("Continue progress to global minima")
                    count = 0
                previous_validation_loss = validation_loss 

        # torch.save(self.model.state_dict(), model_path)
    
#     def evaluate(self, test_loader, batch_size=1, n_features=1):
#         with torch.no_grad():
#             predictions = []
#             values = []
#             for x_test, y_test in test_loader:
#                 x_test = x_test.view([batch_size, 5, n_features])
#                 y_test = y_test
#                 self.model.eval()
#                 yhat = self.model(x_test)
#                 predictions.append(yhat.detach().numpy())
#                 values.append(y_test.detach().numpy())

#         return predictions, values
    
    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()


def get_model(model, model_params):
    models = {"lstm": LSTMModel}
    return models.get(model.lower())(**model_params)


import torch.optim as optim

input_dim = trainX_lat.shape[2]
output_dim = 1
hidden_dim = 32
layer_dim = 1
batch_size = batch_size
dropout = 0
n_epochs = n_epochs
learning_rate = 1e-4
weight_decay = 1e-6

model_params = {'input_dim': input_dim,
                'hidden_dim' : hidden_dim,
                'layer_dim' : layer_dim,
                'output_dim' : output_dim,
                'dropout_prob' : dropout }

model = get_model('lstm', model_params)

loss_fn = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

opt_lat = Optimization_Lat(model=model, loss_fn=loss_fn, optimizer=optimizer)
opt_lat.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
opt_lat.plot_losses()

# predictions, values = opt.evaluate(test_loader_one, batch_size=1, n_features=input_dim)



model_lat = torch.load('model_lat')


y_pred_lat = model_lat(testX_lat)


y_pred_lat = y_pred_lat.detach().numpy()



y_pred_lat_origin = scaler_lat.inverse_transform(y_pred_lat)

testY_lat = testY_lat.detach().numpy()


y_test_lat_origin = scaler_lat.inverse_transform(testY_lat)


# import MSE for accurarcy metrics
from sklearn.metrics import mean_squared_error
import math

print(mean_squared_error(y_test_lat_origin, y_pred_lat_origin))
print(math.sqrt(mean_squared_error(y_test_lat_origin, y_pred_lat_origin)))


# ### Longitude Prediction


cols_target_long = list(df_WP_2018_MANGKHUT)[2:3]


print(cols_features)
print(cols_target_long)


for i in range(len(tc_name_list)):
    for name in tc_name_list[i]:
        globals()['df_features_long_%s_%s'%((str(2000+i)), name)] = globals()['df_WP_%s_%s'%((str(2000+i)), name)][cols_features].astype(float)
        globals()['df_target_long_%s_%s'%((str(2000+i)), name)] = globals()['df_WP_%s_%s'%((str(2000+i)), name)][cols_target_long].astype(float)


trainX_long = []
trainY_long = []
valX_long = []
valY_long = []
testX_long = []
testY_long = []



for i in range(len(tc_name_list)): ### 20 years of data for training (2000-2020)
    for name in tc_name_list[i]:
        globals()['df_features_long_%s_%s'%((str(2000+i)), name)] = globals()['df_features_long_%s_%s'%((str(2000+i)), name)].to_numpy()
        globals()['df_target_long_%s_%s'%((str(2000+i)), name)] = globals()['df_target_long_%s_%s'%((str(2000+i)), name)].to_numpy()


n_future = 1
n_past = 5

for i in range(len(tc_name_list)):
    for name in tc_name_list[i]:
            for j in range(n_past, len(globals()['df_features_long_%s_%s'%((str(2000+i)), name)])-n_future+1):
                trainX_long.append(globals()['df_features_long_%s_%s'%((str(2000+i)), name)][j-n_past:j,0:globals()['df_features_long_%s_%s'%((str(2000+i)), name)].shape[1]])
                trainY_long.append(globals()['df_target_long_%s_%s'%((str(2000+i)), name)][j+n_future-1:j+n_future,0])


trainX_long, trainY_long= np.array(trainX_long), np.array(trainY_long)
valX_long, valY_long= np.array(valX_long), np.array(valY_long)
testX_long, testY_long= np.array(testX_long), np.array(testY_long)

# shuffle_idx
# import random
# shuffle_idx= random.shuffle([i for i in range(10000)])
# trainX = trainX[shuffle_idx]
# trainY = trainY[shuffle_idx]


print(trainX_long.shape)
print(trainY_long.shape)
print(valX_long.shape)
print(valY_long.shape)
print(testX_long.shape)
print(testY_long.shape)


trainX_long, valX_long, trainY_long, valY_long = train_test_split(trainX_long,trainY_long,test_size=0.20, random_state=42)
valX_long, testX_long, valY_long, testY_long = train_test_split(valX_long,valY_long,test_size=0.50, random_state=42)


print(trainX_long.shape)
print(trainY_long.shape)
print(valX_long.shape)
print(valY_long.shape)
print(testX_long.shape)
print(testY_long.shape)

trainX_long.shape[1]

trainX_long = torch.from_numpy(trainX_long).float()
trainY_long = torch.from_numpy(trainY_long).float()
valX_long = torch.from_numpy(valX_long).float()
valY_long = torch.from_numpy(valY_long).float()
testX_long = torch.from_numpy(testX_long).float()
testY_long = torch.from_numpy(testY_long).float()

from torch.utils.data import TensorDataset, DataLoader
batch_size = 32
n_features = 17
n_epochs = 50

train = TensorDataset(trainX_long, trainY_long)
val = TensorDataset(valX_long, valY_long)
test = TensorDataset(testX_long, testY_long)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, drop_last=True)
# test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
# test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)


class Optimization_Long:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, x, y):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()
    
    def train(self, train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=n_features):   
        model_path = "model_lon"
        validation_list = []
        previous_validation_loss = 100
        patience = 4
        count = 0
        
        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, 5, n_features])
                y_batch = y_batch
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, 5, n_features])
                    y_val = y_val
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)
                print(f"[{epoch}/{n_epochs}] Training loss: {training_loss:.8f}\t Validation loss: {validation_loss:.8f}")
                validation_list.append(validation_loss)
                validation_list.sort()
                if validation_loss <= validation_list[0]:
                    torch.save(self.model, model_path)
                    print("Updated best model")
                if validation_loss > previous_validation_loss:
                    count +=1
                    print("Current Validation_loss is greater than Previous Validation Loss, counting:", count)
                    if count>=patience:
                        print("Early stopping executed!")
                        break
                else:
                    print("Continue progress to global minima")
                    count = 0
                previous_validation_loss = validation_loss 

        # torch.save(self.model.state_dict(), model_path)
    
#     def evaluate(self, test_loader, batch_size=1, n_features=1):
#         with torch.no_grad():
#             predictions = []
#             values = []
#             for x_test, y_test in test_loader:
#                 x_test = x_test.view([batch_size, 5, n_features])
#                 y_test = y_test
#                 self.model.eval()
#                 yhat = self.model(x_test)
#                 predictions.append(yhat.detach().numpy())
#                 values.append(y_test.detach().numpy())

#         return predictions, values
    
    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()


import torch.optim as optim

input_dim = trainX_long.shape[2]
output_dim = 1
hidden_dim = 128
layer_dim = 1
batch_size = batch_size
dropout = 0
n_epochs = n_epochs
learning_rate = 0.001
weight_decay = 1e-6

model_params = {'input_dim': input_dim,
                'hidden_dim' : hidden_dim,
                'layer_dim' : layer_dim,
                'output_dim' : output_dim,
                'dropout_prob' : dropout}

model_long = get_model('lstm', model_params)

loss_fn = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model_long.parameters(), lr=learning_rate, weight_decay=weight_decay)

opt_lon = Optimization_Long(model=model_long, loss_fn=loss_fn, optimizer=optimizer)
opt_lon.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
opt_lon.plot_losses()

# predictions, values = opt.evaluate(test_loader_one, batch_size=1, n_features=input_dim)


model_long = torch.load('model_lon')


y_pred_long = model_long(testX_long)


y_pred_long = y_pred_long.detach().numpy()


y_pred_long_origin = scaler_long.inverse_transform(y_pred_long)


testY_long = testY_long.detach().numpy()


y_test_long_origin = scaler_long.inverse_transform(testY_long)


print(mean_squared_error(y_test_long_origin, y_pred_long_origin))
print(math.sqrt(mean_squared_error(y_test_long_origin, y_pred_long_origin)))



# ### Latitude Prediction (Mangkhut)


df_features_lat_2018_MANGKHUT = df_WP_2018_MANGKHUT[cols_features].astype(float)
df_target_lat_2018_MANGKHUT = df_WP_2018_MANGKHUT[cols_target].astype(float)


df_features_lat_2018_MANGKHUT = df_features_lat_2018_MANGKHUT[cols_features].to_numpy()
df_target_lat_2018_MANGKHUT = df_target_lat_2018_MANGKHUT[cols_target].to_numpy()


X_original_MANGKHURT_lat=[]
Y_original_MANGKHURT_lat=[]


for i in range(n_past, len(df_features_lat_2018_MANGKHUT)-n_future+1):
    X_original_MANGKHURT_lat.append(df_features_lat_2018_MANGKHUT[i-n_past:i,0:df_features_lat_2018_MANGKHUT.shape[1]])
    Y_original_MANGKHURT_lat.append(df_target_lat_2018_MANGKHUT[i+n_future-1:i+n_future,0])

X_original_MANGKHURT_lat, Y_original_MANGKHURT_lat= np.array(X_original_MANGKHURT_lat), np.array(Y_original_MANGKHURT_lat)



X_original_MANGKHURT_lat.shape



X_original_MANGKHURT_lat = torch.from_numpy(X_original_MANGKHURT_lat).float()


y_pred_lat_MANGKHURT = model_lat(X_original_MANGKHURT_lat)


y_pred_lat_MANGKHURT  = y_pred_lat_MANGKHURT.detach().numpy()


# forecast_copies_MANGKHURT = np.repeat(forecast_lat_MANGKHURT, df_target_lat_2018_MANGKHUT.shape[1], axis=-1)
# y_pred_lat_MANGKHURT = scaler_lat.inverse_transform(forecast_copies_MANGKHURT)[:,0]
y_pred_lat_MANGKHURT = scaler_lat.inverse_transform(y_pred_lat_MANGKHURT)


y_pred_lat_MANGKHURT.tolist()
y_pred_lat_MANGKHURT_formatted = [ '%.2f' % elem for elem in y_pred_lat_MANGKHURT ]
y_pred_lat_MANGKHURT_formatted = [float(i) for i in y_pred_lat_MANGKHURT_formatted]
print(y_pred_lat_MANGKHURT_formatted)
print(len(y_pred_lat_MANGKHURT_formatted))


Y_original_MANGKHURT_lat = scaler_lat.inverse_transform(Y_original_MANGKHURT_lat)


# import MSE for accurarcy metrics
from sklearn.metrics import mean_squared_error
import math

print(mean_squared_error(Y_original_MANGKHURT_lat, y_pred_lat_MANGKHURT))
print(math.sqrt(mean_squared_error(Y_original_MANGKHURT_lat, y_pred_lat_MANGKHURT)))


# ### Longitude Prediction (Mangkhut)


df_features_long_2018_MANGKHUT = df_WP_2018_MANGKHUT[cols_features].astype(float)
df_target_long_2018_MANGKHUT = df_WP_2018_MANGKHUT[cols_target_long].astype(float)


df_features_long_2018_MANGKHUT = df_features_long_2018_MANGKHUT[cols_features].to_numpy()
df_target_long_2018_MANGKHUT = df_target_long_2018_MANGKHUT[cols_target_long].to_numpy()

X_original_MANGKHURT_long=[]
Y_original_MANGKHURT_long=[]

for i in range(n_past, len(df_features_long_2018_MANGKHUT)-n_future+1):
    X_original_MANGKHURT_long.append(df_features_long_2018_MANGKHUT[i-n_past:i,0:df_features_long_2018_MANGKHUT.shape[1]])
    Y_original_MANGKHURT_long.append(df_target_long_2018_MANGKHUT[i+n_future-1:i+n_future,0])


X_original_MANGKHURT_long, Y_original_MANGKHURT_long= np.array(X_original_MANGKHURT_long), np.array(Y_original_MANGKHURT_long)


X_original_MANGKHURT_long = torch.from_numpy(X_original_MANGKHURT_long).float()


y_pred_long_MANGKHURT = model_long(X_original_MANGKHURT_long)


y_pred_long_MANGKHURT  = y_pred_long_MANGKHURT.detach().numpy()


y_pred_long_MANGKHURT = scaler_long.inverse_transform(y_pred_long_MANGKHURT)

y_pred_long_MANGKHURT.tolist()
y_pred_long_MANGKHURT_formatted = [ '%.2f' % elem for elem in y_pred_long_MANGKHURT ]
y_pred_long_MANGKHURT_formatted = [float(i) for i in y_pred_long_MANGKHURT_formatted]
print(y_pred_long_MANGKHURT_formatted)
print(len(y_pred_long_MANGKHURT_formatted))


Y_original_MANGKHURT_long = scaler_long.inverse_transform(Y_original_MANGKHURT_long)


print(mean_squared_error(Y_original_MANGKHURT_long, y_pred_long_MANGKHURT))
print(math.sqrt(mean_squared_error(Y_original_MANGKHURT_long, y_pred_long_MANGKHURT)))


# ### Prediction visualisation (MANGKHURT)

# Y_original_MANGKHURT_lat = scaler_lat_target.inverse_transform(Y_original_MANGKHURT_lat)


# Y_original_MANGKHURT_long = scaler_long_target.inverse_transform(Y_original_MANGKHURT_long)


print("Actual Lat of Mangkhurt:")
Y_original_MANGKHURT_lat.tolist()
Y_original_MANGKHURT_lat_formatted = [ '%.2f' % elem for elem in Y_original_MANGKHURT_lat ]
Y_original_MANGKHURT_lat_formatted = [float(i) for i in Y_original_MANGKHURT_lat_formatted]
print(Y_original_MANGKHURT_lat_formatted)
print(len(Y_original_MANGKHURT_lat_formatted))
print("Actual Long of Mangkhurt:")

Y_original_MANGKHURT_long.tolist()
Y_original_MANGKHURT_long_formatted = [ '%.2f' % elem for elem in Y_original_MANGKHURT_long ]
Y_original_MANGKHURT_long_formatted = [float(i) for i in Y_original_MANGKHURT_long_formatted]
print(Y_original_MANGKHURT_long_formatted)
print(len(Y_original_MANGKHURT_long_formatted))


print("Predicted Lat of Mangkhurt:")
print(y_pred_lat_MANGKHURT_formatted)
print("----------------------------------------------------------------------------------------------")
print("Predicted Long of Mangkhurt:")
print(y_pred_long_MANGKHURT_formatted)


import os
os.environ['PROJ_LIB'] = r'C:\ProgramData\Anaconda3\pkgs\proj4-5.2.0-h6538335_1006\Library\share'
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.path as mpath


def get_hurricane():
    u = np.array([  [2.444,7.553],
                    [0.513,7.046],
                    [-1.243,5.433],
                    [-2.353,2.975],
                    [-2.578,0.092],
                    [-2.075,-1.795],
                    [-0.336,-2.870],
                    [2.609,-2.016]  ])
    u[:,0] -= 0.098
    codes = [1] + [2]*(len(u)-2) + [2] 
    u = np.append(u, -u[::-1], axis=0)
    codes += codes

    return mpath.Path(3*u, codes, closed=False)


hurricane = get_hurricane()


plt.figure(figsize=(20,10))
m = Basemap(projection='mill', llcrnrlat = 0, urcrnrlat = 30, llcrnrlon = 90, urcrnrlon =175, resolution = 'c')
m.drawcoastlines()
m.drawparallels(np.arange(-90,90,10), labels=[True,False,False,False])
m.drawmeridians(np.arange(-180,180,30), labels=[0,0,0,1])

m.scatter(y_pred_long_MANGKHURT_formatted, y_pred_lat_MANGKHURT_formatted, latlon=True, s=100, c='blue', marker=hurricane, label = 'Louis model predicted track')
m.scatter(Y_original_MANGKHURT_long, Y_original_MANGKHURT_lat, latlon=True, s=100, c='red', marker=hurricane, label = 'Actual Mangkhurt Track')
m.scatter(114.1694, 22.3193,latlon=True, s=200, c='orange', marker='o', label = 'Hong Kong' )
plt.title('Typhoon track prediction (Mangkhurt)' , fontsize=20)
plt.legend(prop={'size': 20})
plt.savefig("1st prediction_4th.png")
plt.show()

