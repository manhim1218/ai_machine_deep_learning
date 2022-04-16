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
from skimage import io, transform
import os
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image


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
        globals()['df_WP_%s_%s'%((str(2000+i)), name)].rename(columns={'ISO_TIME':'Date'}, inplace=True)
        globals()['df_WP_%s_%s'%((str(2000+i)), name)]['Date'] = pd.to_datetime(globals()['df_WP_%s_%s'%((str(2000+i)), name)]['Date'], dayfirst=True)


cols_features = list(df_WP_2018_MANGKHUT)[1:len(df_WP_2018_MANGKHUT.columns)]
print(cols_features)



cols_target = list(df_WP_2018_MANGKHUT)[1:2]



print(cols_features)
print(cols_target)

# for i in range(len(tc_name_list)):
#     for name in tc_name_list[i]:
#         globals()['df_features_lat_%s_%s'%((str(2000+i)), name)] = globals()['df_WP_%s_%s'%((str(2000+i)), name)][cols_features].astype(float)
#         globals()['df_target_lat_%s_%s'%((str(2000+i)), name)] = globals()['df_WP_%s_%s'%((str(2000+i)), name)][cols_target].astype(float)


count = 0

def get_typhoon_data(typhoon_name, typhoon_year, idx , n_past, n_future):
    global count
    features_list = []
    image_list = []
    count_photo =[]
    image_list_current = []
    for i in range(n_past):
        features_list_current = []
        image_list_current = []
        root_dir = "C:/Users/Louis/Full Typhoon Images/"
        # current_typhoon_name = df_WP_2000_DAMREY.iloc[idx,1]
        # current_timestamp_year = (str(df_WP_2000_DAMREY.iloc[idx,0]))[:4]
        folder = "(" + typhoon_name + ")_" + typhoon_year
        current_timestamp_date = (str(globals()['df_WP_%s_%s'% (typhoon_year, typhoon_name)].iloc[idx,0]))[:10]
        current_timestamp_hour = (str(globals()['df_WP_%s_%s'% (typhoon_year, typhoon_name)].iloc[idx,0]))[11:13]
        image_path = "(" + typhoon_name + ")_" + current_timestamp_date + "_" + current_timestamp_hour + ".png"
        img_name = os.path.join(root_dir, folder ,image_path).replace("\\","/")
        # print(img_name)
        try:
            # image_test = io.imread(img_name)
            scaler = transforms.Scale((224, 224))
            # normalize = transforms.Normalize((0.5), (0.5))
            normalize = transforms.Normalize((0.5), (0.5))
            to_tensor = transforms.ToTensor()
            
            image_test = Image.open(img_name)
            image = Variable(normalize(to_tensor(scaler(image_test))).unsqueeze(0))
            
            # image_test = Image.open(img_name).convert('RGB')
            # image_test = np.array(image_test)
            # image_test = np.rollaxis(image_test, 2, 0)
        
            print(image.shape)
            # current_image_vector = get_vector(img_name)
            # current_image_vector = current_image_vector.numpy()
            print("OK")
            count = count+1
            print(count)
            # print(typhoon_name)
            # print(typhoon_year)
            # print("read image")
        except FileNotFoundError:
            print("FileNotFoundError")
            # print(typhoon_name)
            # print(typhoon_year)
            # print(current_timestamp_date)
            # print("No satelite image for this timestamp")
            # image = "no_image"
            image = torch.zeros(1, 1,224,224)
            print(image.shape)

        for j in range(17):
            features_list_current.append(globals()['df_WP_%s_%s'% (typhoon_year, typhoon_name)].iloc[idx,j+2])
        image_list_current.append(image)
        
        features_list.append(features_list_current)
        image_list.append(image_list_current)
    
        # features_list = np.array(features_list, dtype=np.float32)
        # image_list = np.array(image_list, dtype=np.float32)
        
        # all_features_numpy = np.concatenate((features_list_current, current_image_vector), axis=0)
        # feature_list.append(all_features_numpy)
        # image_list.append(typhoon_image_current_timestamp)
        # typhoon_dict['features'].append(features_list_current)
        # typhoon_dict['typhoon_image'].append(typhoon_image_current_timestamp)
        idx += 1
    target_latitude = globals()['df_WP_%s_%s'% (typhoon_year, typhoon_name)].iloc[idx,2]
    target_longitude =  globals()['df_WP_%s_%s'% (typhoon_year, typhoon_name)].iloc[idx,3]
    return features_list, image_list, target_latitude, target_longitude,count


n_past = 5
n_future = 1

all_features_typhoon_data=[]
all_target_typhoon_lat=[]
all_target_typhoon_long=[]
# for i in range(len(tc_name_list)):
for i in range(1):
    for typhoon_name in tc_name_list[i]:
        idx=0
        typhoon_year = str(2000 + i)
        print(typhoon_year)
        print(typhoon_name)
        while True:
            try:
                list_1, list_2, lat,long, count = get_typhoon_data(typhoon_name, typhoon_year, idx , n_past, n_future)
                features_tensor = torch.FloatTensor(list_1)
                image_path = list_2
                lat_tensor = torch.from_numpy(np.array(lat, dtype="float32"))
                long_tensor = torch.from_numpy(np.array(long, dtype="float32"))

                
                # print("doing")
                typhoon_features_dict = {'features': features_tensor,
                                'images': image_path}
                
                typhoon_target_lat = {'latitude':lat_tensor}
                typhoon_target_long = {'longitude':long_tensor}

                all_features_typhoon_data.append(typhoon_features_dict)
                all_target_typhoon_lat.append(typhoon_target_lat)
                all_target_typhoon_long.append(typhoon_target_long)
                
                idx += 1
                list_1 = []
                list_2 = []
                lat = 0 
                long = 0
            except IndexError:
                print("no more timestamp")
                break


from sklearn.model_selection import train_test_split

trainX_lat, valX_lat, trainY_lat, valY_lat = train_test_split(all_features_typhoon_data,all_target_typhoon_lat,test_size=0.10, random_state=42)
# valX_lat, testX_lat, valY_lat, testY_lat = train_test_split(valX_lat,valY_lat,test_size=0.50, random_state=42)



train_set_with_lat = [trainX_lat, trainY_lat]


val_set_with_lat = [valX_lat, valY_lat]


from torch.utils.data import Dataset,DataLoader
import cv2
import tensorflow as tf


# Initialize the model
model = models.vgg16(pretrained=True)
new_model = FeatureExtractor(model)


class CustomTensorDataset(Dataset):
  def __init__(self, dataset, transform_list=None):
    [data_X, data_y] = dataset
    #X_tensor, y_tensor = Tensor(data_X), Tensor(data_y)
    tensors = (data_X, data_y)
    assert all(len(tensors[0]) == len(tensor) for tensor in tensors)
    self.tensors = tensors
    
  def __getitem__(self, index):
    x = self.tensors[0][index]
    y = self.tensors[1][index]
    
    tmp_img_data = []
    images_data = []
    
    for image_name in x['images']:
        # Transform the image, so it becomes readable with the model
        
        # transform = transforms.Compose([
        #   transforms.ToPILImage(),
        #   transforms.CenterCrop(512),
        #   transforms.Resize(448),
        #   transforms.ToTensor()                              
        # ])
        
        # print(image_name)

        # Will contain the feature
        # features = []
        
        # path = image_name[0]
        
        # img = Image.open(path).convert('RGB')
        
        # img = cv2.imread(path,1)

        
        # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        
        
        # try:
        #     img = transform(img)
        #     img = img.reshape(1, 3, 448, 448)
        # except TypeError:
        #     img = torch.zeros(3, 448, 448)
        #     img = img.reshape(1, 3, 448, 448)
        
        tmp_img_data.append(image_name)
        
    images_data.append(tmp_img_data)
    # images_data = torch.FloatTensor(images_data)
        
    #     with torch.no_grad():
    #         feature = new_model(img)
    #         features.append(feature)
    # images_features = features[0]

    
#     print(image_name[0])
#     1. Load the image with Pillow library
#     img = io.imread(image_name[0])
#     print(img)
#     2. Create a PyTorch Variable with the transformed image


#     normalize = transforms.Normalize((0.5),(0.5))
#     to_tensor = transforms.ToTensor()
#     scaler = transforms.Scale((224, 224))


#     img = Image.open(image_name[0]).convert('RGB')
#     # print(img)
#     t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
#     images.append(t_img)
    
    inputs = {
        'features': x['features'],
        'images': images_data
    }
        
    # print(x)
    # print(y)
    
    return inputs, y

  def __len__(self):
    return len(self.tensors[0])


train_set_lstm = CustomTensorDataset(dataset=train_set_with_lat)


train_set_lstm.__getitem__(3)

val_set_lstm = CustomTensorDataset(dataset=val_set_with_lat)


# ### DataLoader from dictionary


batch_size = 1
# n_features = len(all_features_typhoon_data[0]['features'])
n_epochs = 100


train_loader = DataLoader(train_set_lstm, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(val_set_lstm, batch_size=batch_size, shuffle=False, drop_last=True)
# test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=True)


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
        
        # split 529 features into 17 + 512
        # 512 --> 32 dimensions

        # positional encoding
        
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
        #self.resnet = image_ex_model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []


    def Resnet(self,image):
        model_resnet = models.resnet18(pretrained=True)
        model_resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        layer = model_resnet._modules.get('avgpool')
        
        # print(t_img)
        # 3. Create a vector of zeros that will hold our feature vector
        #    The 'avgpool' layer has an output size of 512
        my_embedding = torch.zeros(512)
        # 4. Define a function that will copy the output of a layer
        def copy_data(m, i, o):
            my_embedding.copy_(o.data.reshape(o.data.size(1)))
        # 5. Attach that function to our selected layer
        h = layer.register_forward_hook(copy_data)
        # 6. Run the model on our transformed image
        model_resnet(image)
        # 7. Detach our copy function from the layer
        h.remove()
        # 8. Return the feature vector
        return my_embedding
    
    def tensor_concat(self,typhoon_features, typhoon_images, y):
        
        image_feature_model = []
        for i in range(len(typhoon_images[0])):
            image_Feature = self.Resnet(typhoon_images[0][i][0][0,:,:,:,:])
            image_feature_model.append(image_Feature)

        image_tensor = torch.stack((image_feature_model[0], image_feature_model[1], image_feature_model[2], image_feature_model[3], image_feature_model[4]), 0)
        image_tensor = image_tensor[None,:,:]
        
        # print(image_tensor.shape)       
        # print(typhoon_features.shape)

        
        merge_image_and_feature_list = []
        for i in range(len(typhoon_features.data[0])):
            merge_image_and_feature = torch.concat((typhoon_features.data[0][i], image_tensor.data[0][i]), 0)
            # print(merge_image_and_feature.shape)
            merge_image_and_feature_list.append(merge_image_and_feature)
        
        final_tensor = torch.stack((merge_image_and_feature_list[0], merge_image_and_feature_list[1], merge_image_and_feature_list[2], merge_image_and_feature_list[3], merge_image_and_feature_list[4]), 0)
        
        final_tensor = final_tensor[None,:,:]
        # print(final_tensor)
        # print(final_tensor.shape)
        
        return final_tensor
        
    
    def train_step(self, typhoon_features, typhoon_images, y):
        # Sets model to train mode
        self.model.train()
            
        final_tensor = self.tensor_concat(typhoon_features,typhoon_images,y)
        
        y = y[None,:]
#         for i in range(len(typhoon_images[0])):
#             # try:
#             print(typhoon_images[0][i].shape)
#             img = transform(typhoon_images[0][i][0,:,:,:])
#             img = img.reshape(1, 3, 448, 448)
#             # except TypeError:
#                 # img = torch.zeros(3, 448, 448)
#                 # img = img.reshape(1, 3, 448, 448)
#             with torch.no_grad():
#                 # print(typhoon_images[0][i])
#                 feature = new_model(img)
#                 print(feature)
#                 print(feature.shape)
#                 image_feature_model.append(feature)
        
        
#         last_tensor = image_feature_model[0]
#         for i in range(len(image_feature_model)):
#             try:
#                 final_image_tensor = torch.concat((last_tensor, image_feature_model[i+1]), 0)
#                 last_tensor = final_image_tensor
#             except IndexError: 
#                 break

#         print(y['latitude'])
        
        # image_Feature = self.Resnet(x['images'][0,:,:,:,:])

        #     features.append(feature)
        # images_features = features[0]
        
#         final_image_tensor = final_image_tensor[None,:,:]
        
#         print(typhoon_features.shape)
#         print(final_image_tensor.shape)
        
        # x_concat = torch.concat((typhoon_features, final_image_tensor), 0) # 17 + 512
        
        # print(x_concat)
        
#         x0_lat = x[0][0]['features'][0][0]
        
#         # Makes predictions
#         delta_y = self.model(x_concat)
        
#         yhat = x0_lat + delta_y
        
        # Makes predictions
        yhat = self.model(final_tensor)


        #Computes loss
        loss = self.loss_fn(y, yhat)  #yhat + x0 (lat/long)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()
    
    def train(self, train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=17):
        # model_path = f'models/{self.model}_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        
        model_path = "model_lat"
        validation_list = []
        previous_validation_loss = 100
        patience = 4
        count = 0
        
        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                # print(x_batch['features'])
                # print(x_batch['images'])     
                # print('Y batch',y_batch)
                
                x_batch['features'] = x_batch['features'].view([batch_size, 5, 17])
                
                y_batch = y_batch['latitude']
                
                # x_batch = x_batch.view([batch_size, 5, n_features])
                
                # print(x_batch)
                # print(y_batch)
                
                loss = self.train_step(x_batch['features'],x_batch['images'], y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)
            print(training_loss)
            
            
            
            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val['features'] = x_val['features'].view([batch_size, 5, 17])
                    y_val = y_val['latitude']
                    y_val = y_val[None,:]
                    final_tensor = self.tensor_concat(x_val['features'],x_val['images'],y_val)
                    self.model.eval()
                    yhat = self.model(final_tensor)
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

input_dim = 529
output_dim = 1
hidden_dim = 128
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
#Resnet_model = models.resnet18(pretrained=True)


loss_fn = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) #list [resnet.parameters(), model.parameters()]

opt_lat = Optimization_Lat(model=model, loss_fn=loss_fn, optimizer=optimizer)
opt_lat.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
opt_lat.plot_losses()


model_lat = torch.load('model_lat')




