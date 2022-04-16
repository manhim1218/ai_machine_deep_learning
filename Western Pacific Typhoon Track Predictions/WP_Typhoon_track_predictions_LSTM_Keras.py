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


full_WP = full_WP.iloc[1:]


full_WP = full_WP.loc[full_WP['SEASON'].astype(int) >= 2000]


columns_features = ['ISO_TIME', 'SEASON', 'NAME', 'USA_LAT', 'USA_LON', 'DIST2LAND', 'USA_WIND', 'USA_PRES', 'USA_R34_NE', 'USA_R34_SE', 'USA_R34_SW', 'USA_R34_NW', 'USA_R50_NE', 'USA_R50_SE', 'USA_R50_SW', 'USA_R50_NW', 'USA_R64_NE', 'USA_R64_SE', 'USA_R64_SW', 'USA_R64_NW']


# columns_features_trial =  ['ISO_TIME', 'SEASON', 'NAME', 'USA_LAT', 'USA_LON', 'USA_WIND', 'USA_PRES']


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


# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler_lat = MinMaxScaler()
# scaler_long = MinMaxScaler()
# all_features = ['USA_LAT','USA_LON', 'DIST2LAND' ,'USA_WIND','USA_PRES','USA_R34_NE','USA_R34_SE','USA_R34_SW','USA_R34_NW','USA_R50_NE','USA_R50_SE','USA_R50_SW',
#                'USA_R50_NW', 'USA_R64_NE', 'USA_R64_SE', 'USA_R64_SW' ,'USA_R64_NW']
# full_WP_modify[all_features] = scaler.fit_transform(full_WP_modify[all_features])



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

count


# ### Data Preprcoessing


# for name in list_WP_2018:
#     globals()['df_WP_2018_%s' % name] = globals()['df_WP_2018_%s' % name].replace(r'^\s*$', -1 , regex=True)


# for i in range(len(tc_name_list)):
#     for name in tc_name_list[i]:
#          globals()['df_WP_%s_%s'%((str(2000+i)), name)] = (globals()['df_WP_%s_%s'%((str(2000+i)), name)]).replace(r'^\s*$', -1 , regex=True)



# df_WP_2000_BOLAVEN



for i in range(len(tc_name_list)):
    for name in tc_name_list[i]:
        globals()['df_WP_%s_%s'%((str(2000+i)), name)].rename(columns={'ISO_TIME':'Date'}, inplace=True)
        globals()['df_WP_%s_%s'%((str(2000+i)), name)]['Date'] = pd.to_datetime(globals()['df_WP_%s_%s'%((str(2000+i)), name)]['Date'], dayfirst=True)



# df_WP_2018_MANGKHUT



cols_features = list(df_WP_2018_MANGKHUT)[1:len(df_WP_2018_MANGKHUT.columns)]
print(cols_features)


# fig, axes = plt.subplots(nrows=5, ncols=4)
# f = 0
# for i in range(5):
#     for j in range(4):
#         if (f<len(cols_features)):
#             dfMANGKHURT.set_index(dfMANGKHURT['Date'])[cols_features[f]].plot(figsize=(30,30), legend=True, ax=axes[i][j], color='green')
#             f=f+1
#         else:
#             axes[i,j].set_axis_off()


# ### Latitude Prediction



cols_target = list(df_WP_2018_MANGKHUT)[1:2]



print(cols_features)
print(cols_target)


# tc_name_list[2].remove('CHANGMI')
# tc_name_list[4].remove('MERBOK')
# tc_name_list[7].remove('PODUL')


for i in range(len(tc_name_list)):
    for name in tc_name_list[i]:
        globals()['df_features_lat_%s_%s'%((str(2000+i)), name)] = globals()['df_WP_%s_%s'%((str(2000+i)), name)][cols_features].astype(float)
        globals()['df_target_lat_%s_%s'%((str(2000+i)), name)] = globals()['df_WP_%s_%s'%((str(2000+i)), name)][cols_target].astype(float)

df_features_lat_2018_MANGKHUT

# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# for i in range(len(tc_name_list)):
#     for name in tc_name_list[i]:
#         try:
#             globals()['df_features_lat_scaled_%s_%s'%((str(2000+i)), name)] = scaler.fit_transform(globals()['df_features_lat_%s_%s'%((str(2000+i)), name)])
#             globals()['df_target_lat_scaled_%s_%s'%((str(2000+i)), name)] = scaler.fit_transform(globals()['df_target_lat_%s_%s'%((str(2000+i)), name)])
# #             print((str(2000+i)))
# #             print(name)
#         except ValueError:
# #             print((str(2000+i)))
#             print(name)



# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# for i in range(len(tc_name_list)):
#     for name in tc_name_list[i]:
#         try:
#             globals()['df_target_lat_scaled_%s_%s'%((str(2000+i)), name)] = scaler.fit_transform(globals()['df_target_lat_%s_%s'%((str(2000+i)), name)])
#             print(scaler.data_max_())
# #             print((str(2000+i)))
# #             print(name)
#         except ValueError:
# #             print((str(2000+i)))
#             print(name)



# tc_name_list[2].remove('CHANGMI')
# tc_name_list[4].remove('MERBOK')
# tc_name_list[7].remove('PODUL')
# tc_name_list[8].remove('PHANFONE')
# tc_name_list[13].remove('YUTU')
# tc_name_list[14].remove('MITAG')
# tc_name_list[16].remove('MALOU')
tc_name_list[18].remove('MANGKHUT')

trainX_lat = []
trainY_lat = []

for i in range(len(tc_name_list)):
    for name in tc_name_list[i]:
        globals()['df_features_lat_%s_%s'%((str(2000+i)), name)] = globals()['df_features_lat_%s_%s'%((str(2000+i)), name)].to_numpy()
        globals()['df_target_lat_%s_%s'%((str(2000+i)), name)] = globals()['df_target_lat_%s_%s'%((str(2000+i)), name)].to_numpy()


n_future = 1
n_past = 5

for i in range(len(tc_name_list)):
    for name in tc_name_list[i]:
            for j in range(n_past, len(globals()['df_features_lat_%s_%s'%((str(2000+i)), name)])-n_future+1):
#                 print((str(2000+i)))
#                 print(name)
                trainX_lat.append(globals()['df_features_lat_%s_%s'%((str(2000+i)), name)][j-n_past:j,0:globals()['df_features_lat_%s_%s'%((str(2000+i)), name)].shape[1]])
                trainY_lat.append(globals()['df_target_lat_%s_%s'%((str(2000+i)), name)][j+n_future-1:j+n_future,0])



trainX_lat, trainY_lat= np.array(trainX_lat), np.array(trainY_lat)
# shuffle_idx
# import random
# shuffle_idx= random.shuffle([i for i in range(10000)])
# trainX = trainX[shuffle_idx]
# trainY = trainY[shuffle_idx]

print(trainX_lat.shape)
print(trainY_lat.shape)


trainX_lat.shape[1]


# from sklearn.preprocessing import MinMaxScaler
# scaler_lat_features = MinMaxScaler()
# scaler_lat_target = MinMaxScaler()
# trainX_lat = scaler_lat_features.fit_transform(trainX_lat.reshape(-1, trainX_lat.shape[-1])).reshape(trainX_lat.shape)
# trainY_lat = scaler_lat_target.fit_transform(trainY_lat)



# trainY_lat = scaler_lat.fit_transform(trainY_lat)


# ### KERAS LSTM


import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM 
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


# def a function for plotting history data
def plot_history(history):
    # Plot the results (shifting validation curves appropriately)
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'],'b')
    plt.plot(history.history['val_loss'],'g')
    plt.title('Val_loss vs Train_loss')
    plt.xlabel('Number of Epoch')
    plt.ylabel('MSE')
    plt.legend(['Train_loss','Val_loss'])
    plt.grid(True)
    plt.show() 


early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) 
model_checkpoint_lat = ModelCheckpoint('best_model_lat', monitor='val_loss', mode='min', verbose=1, save_best_only=True)



model_lat = Sequential()
model_lat.add(LSTM(64, activation='relu',input_shape=(trainX_lat.shape[1],trainX_lat.shape[2]), return_sequences = False))
# model_lat.add(Dropout(0.2))
model_lat.add(Dense(trainY_lat.shape[1]))
model_lat.compile(optimizer='adam', loss='mse')
model_lat.summary()
history_lat = model_lat.fit(trainX_lat, trainY_lat, epochs=30, batch_size=16, callbacks = [early_stopping_cb, model_checkpoint_lat], validation_split=0.1, verbose=2)
plot_history(history_lat)



best_model_lat = load_model('best_model_lat')


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


y_pred_lat_MANGKHURT = best_model_lat.predict(X_original_MANGKHURT_lat)


y_pred_lat_MANGKHURT.shape


# forecast_copies_MANGKHURT = np.repeat(forecast_lat_MANGKHURT, df_target_lat_2018_MANGKHUT.shape[1], axis=-1)
# y_pred_lat_MANGKHURT = scaler_lat.inverse_transform(forecast_copies_MANGKHURT)[:,0]
y_pred_lat_MANGKHURT = scaler_lat.inverse_transform(y_pred_lat_MANGKHURT)


y_pred_lat_MANGKHURT
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


for i in range(len(tc_name_list)):
    for name in tc_name_list[i]:
        globals()['df_features_long_%s_%s'%((str(2000+i)), name)] = globals()['df_features_long_%s_%s'%((str(2000+i)), name)].to_numpy()
        globals()['df_target_long_%s_%s'%((str(2000+i)), name)] = globals()['df_target_long_%s_%s'%((str(2000+i)), name)].to_numpy()


n_future = 1
n_past = 5

for i in range(len(tc_name_list)):
    for name in tc_name_list[i]:
            for j in range(n_past, len(globals()['df_features_long_%s_%s'%((str(2000+i)), name)])-n_future+1):
#                 print((str(2000+i)))
#                 print(name)
                trainX_long.append(globals()['df_features_long_%s_%s'%((str(2000+i)), name)][j-n_past:j,0:globals()['df_features_long_%s_%s'%((str(2000+i)), name)].shape[1]])
                trainY_long.append(globals()['df_target_long_%s_%s'%((str(2000+i)), name)][j+n_future-1:j+n_future,0])



trainX_long, trainY_long= np.array(trainX_long), np.array(trainY_long)
# shuffle_idx
# import random
# shuffle_idx= random.shuffle([i for i in range(10000)])
# trainX = trainX[shuffle_idx]
# trainY = trainY[shuffle_idx]



# scaler_long_features = MinMaxScaler()
# scaler_long_target = MinMaxScaler()
# trainX_long = scaler_long_features.fit_transform(trainX_long.reshape(-1, trainX_long.shape[-1])).reshape(trainX_long.shape)
# trainY_long = scaler_long_target.fit_transform(trainY_long)



print(trainX_long.shape)
print(trainY_long.shape)


# ### KERAS LSTM


model_checkpoint_long = ModelCheckpoint('best_model_long', monitor='val_loss', mode='min', verbose=1, save_best_only=True)


model_long = Sequential()
model_long.add(LSTM(32, activation='relu',input_shape=(trainX_long.shape[1],trainX_long.shape[2]), return_sequences = False))
model_long.add(Dense(trainY_long.shape[1]))
model_long.compile(optimizer='adam', loss='mse')
model_long.summary()
history_long = model_long.fit(trainX_long, trainY_long, epochs=30, batch_size=16, callbacks = [early_stopping_cb,model_checkpoint_long], validation_split=0.1, verbose=2)
plot_history(history_long)


# ### Longitude prediction (MANGKHURT)


best_model_long = load_model('best_model_long')

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


y_pred_long_MANGKHURT = best_model_long.predict(X_original_MANGKHURT_long)



y_pred_long_MANGKHURT = scaler_long.inverse_transform(y_pred_long_MANGKHURT)

y_pred_long_MANGKHURT
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
os.environ["PROJ_LIB"] = "C:\\Utilities\\Python\\Anaconda\\Library\\share"; #fixr
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

