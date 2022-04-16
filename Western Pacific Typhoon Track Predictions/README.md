# Western Pacific Typhoon trajetories prediction. 

Predicting tropical cyclones trajectories using Convolutionary Neural network and Long Short Term Memory Neural Network 


### General description: 

Two separate LSTM models were established to predict the latitude and longitude of a typhoon track every 3 hour from its previous timstamp. I have utilised Keras and Pytorch deep learning framework separately in Python as to train the LSTM models with 20 years of Western Pacific Region's typhoon data which is publicly available on NOAA data archive. The models were tested with the supertyphoon Mangkhut which was an unseen data to the LSTM model.


### Last progress summary (ie. December 2021)
#### Deep learning framework: Keras
```
Meteorological features: 
Latitude, Longitude, Distance to Land, Wind ,Pressure, 
Radius of maximum wind in 34 miles (NE), Radius of maximum wind in 34 miles (SE), 
Radius of maximum wind in 34 miles (SW), Radius of maximum wind in 34 miles (NW), 
Radius of maximum wind in 50 miles (NE), Radius of maximum wind in 50 miles (SE), 
Radius of maximum wind in 50 miles (SW), Radius of maximum wind in 50 miles (NW), 
Radius of maximum wind in 64 miles (NE), Radius of maximum wind in 64 miles (SE), 
Radius of maximum wind in 64 miles (SW), Radius of maximum wind in 64 miles (NW) 

Cost function (Performance metric): 
RMSE between actual track and predicted track

Latitude RMSE: 
10 km (Good result)

Longitude RMSE: 
65 km (Continue to work on the longitude model to reduce RMSE for longitude predictions)
```

### Current progress (up to April 2022)
#### Deep learning framework: Pytorch
```
Conducted a granger causality test for features selection,however, 
none of the meteorological features had granger causality relationship to the latitude and longitude data. 

Utilised a pretrained Resnet18 Convolution Neural Network to extra features from typhoon satelite images. 
The pretained CNN model delivered a vector representing the feature of the image of each typhoon for each timestamp. 
The extracted image vector was then concantenated with the meteorological features 
```
