# Bike Sharing Demand Predictions

## Overview
This repository contains code and data for predicting bike sharing demand. The goal is to forecast the number of bikes rented in a given time period based on various features such as weather conditions, time of day, and day of the week.

## Dataset
The dataset used for this project contains historical data of bike rentals, including features such as:
- *datetime*: Date and time of the rental
- *season*: Season of the year (spring, summer, fall, winter)
- *holiday*: Whether it's a holiday or not
- *weather*: Weather condition (clear, mist, light rain, heavy rain)
- *temp*: Temperature in Celsius
- *humidity*: Humidity level
- *windspeed*: Wind speed
- *count*: Number of bikes rented

## Setup
To run the code in this repository, follow these steps:
   

1. Navigate to the project directory:
   
   cd bike-sharing-demand
   

2. Install the required dependencies:
   
   pip install -r requirements.txt
   

3. Run the Jupyter notebook or Python scripts to train models and make predictions.

## Models
Various machine learning models are implemented for predicting bike sharing demand, including:
- Linear Regression
- Random Forest
- Gradient Boosting
- Neural Networks

## Usage
1. *Training*: Use the provided Jupyter notebook or Python scripts to train the models on the dataset.

2. *Prediction*: After training, you can use the trained models to make predictions on new data. Simply input the features of the new data into the trained model to get the predicted bike rental count.

3. *Evaluation*: Evaluate the performance of the models using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

## Contributions
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, feel free to open an issue or create a pull request.

## Acknowledgments
Special thanks to the bike sharing company for providing the dataset used in this project.

## Author
© ***[INNOCENT UDO](https://github.com/Yahiamostafa11)***
