# Western Pacific Typhoon trajetories prediction

Predicting the Western Pacific Region tropical cyclones trajectories using various Deep Learning Approaches


### General description: 

This project proposed different deep learning frameworks on real historical typhoon data extracted from the NOAA archive and the Kitamoto laboratory data repository. Various deep learning approaches were conducted to predict the next 3 hours and 6 hours of typhoon tracks in the Western Pacific Region. The experimental result showed that the LSTM regression model and the CNN-LSTM ensemble model delivered a similar prediction result toward the 3-hour prediction, however, the CNN-LSTM ensemble model performed better on 6 hours prediction compared to other deep learning approaches conducted in this project. The CNN-LSTM ensemble model could be a potential deep learning approach to deliver a more robust prediction result if diverse image data type was used to extract relevant track feature in the model.

### Last progress summary (ie. December 2021)
#### Deep learning framework: Keras
```
1. Meteorological features: 
Latitude, Longitude, Distance to Land, Wind ,Pressure, 
Radius of maximum wind in 34 miles (NE), Radius of maximum wind in 34 miles (SE), 
Radius of maximum wind in 34 miles (SW), Radius of maximum wind in 34 miles (NW), 
Radius of maximum wind in 50 miles (NE), Radius of maximum wind in 50 miles (SE), 
Radius of maximum wind in 50 miles (SW), Radius of maximum wind in 50 miles (NW), 
Radius of maximum wind in 64 miles (NE), Radius of maximum wind in 64 miles (SE), 
Radius of maximum wind in 64 miles (SW), Radius of maximum wind in 64 miles (NW) 

2. Cost function (Performance metric): 
RMSE between actual track and predicted track

3. Latitude RMSE: 
10 km (Good result)

4. Longitude RMSE: 
65 km (Continue to work on the longitude model to reduce RMSE for longitude predictions)
```

### Visualisation of predicting supertyphoon Mangkhurt track in Western Pacific Region
<p align="center"><img src="Visualisation of predicting super tyhoon Mangkhurt.png"\></p>

### Current progress (up to April 2022)
#### Deep learning framework: Pytorch
```
1. Conducted a granger causality test for features selection,however, none of the meteorological features had granger causality relationship to the latitude and longitude data. 
2. Utilised a pretrained Resnet18 Convolution Neural Network to extra features from typhoon satelite images. 
3. The pretained CNN model delivered a vector representing the feature of the image of each typhoon for each timestamp. 
4. The extracted image vector was then concantenated with the meteorological features 

Cost function Result: Working in Progress 

```
### Numerical Data (17 meteorological features)
Full WP dataset.csv is provided in this respository. Original csv file can be downloaded from the NOAA data archive.

### Image Data for extracing image features from pretrained Resnet18 CNN model
A python based webscrapping tool ie.Typhoon_image_scrapping_tool.py is provided in this respository. It might take days to scrap all satelite images from the website. ie Western Pacific typhoon images from the last 20 years
