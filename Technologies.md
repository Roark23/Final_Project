# Final_Project


# Technologies Used

Git Hub

Visual Studio Code

Phyton 3

SQL Database

Kaggle

Tableau

Snipping Tool

## Data Cleaning and Analysis

Data cleansing involves spotting and resolving potential data inconsistencies or errors to improve our data quality.
In this process, we review, analyze, detect, modify, or remove “dirty” data to make our dataset “clean.” Data clensing 
is also called data cleaning or data scrubbing.

The way we check for unique values is to use the Pandas DataFrame's 

Pandas: will be used to clean the data and perform an exploratory analysis. Further analysis is completed using Python.

## Database Storage

Selecting the right data store for our requirements is a key design decision. Reason why we have decided to use
SQL Database as an standard language for accessing and manipulating databases, which provides various benefits to our development, 
such as saving time by automating routine tasks, locating and fixing errors, taking advantage of intelligent support from 
the IDE, and increasing overall productivity.

yfinance: is used to download and display information on our selected stocks

## Machine Learning

Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that we learn, and gradually improving its accuracy. Through the use of statistical methods, algorithms are trained to make classifications or predictions, uncovering key insights within our data mining project. Also, The Evaluation metric measures the quality of the machine learning model. ideally impacting key growth metrics.

The machine learning aspect will take in stock data from the last three years of the dataset and will output prediction prices. The code will run two different forecasting models (ARIMA and Facebook Prophet) and will produce graphs displaying the predictions to compare each of the models.

* import numpy as np: keeps our code more readable

* import pandas as pd: tells Phyton to bring the pandas data analysis library into our current environment

* import matplotlib.pyplot as plt: is importer into our namespace under the shorter name plt. The pyplot is where the plot () scatter(), and other commands live

* from pandas.plotting import lag_plot: is a scatter plot for a time series and the same data lagged

* import datetime as dt: Return a string stating the day of the week corresponding to datetime dt

* from sklearn.linear_model import LinearRegression: Is one of the best statistical models that studies the relationship between a dependent variable (Y) with a given set of independent variables (X). The relationship is established with the help of fitting a best line. sklearn.linear_model.LinearRegression is the module used to implement linear regression.

* from sklearn.model_selection import train_test_split: is a function in Sklearn model selection for splitting data arrays into two subsets: for training data and for testing data. With this function, we do not need to divide the dataset manually. By default, Sklearn train_test_split will make random partitions for the two subsets.

* from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error: Array-like value defines weights used to average errors. ‘raw_values’ : Returns a full set of errors in case of multioutput input. ‘uniform_average’ : Errors of all outputs are averaged with uniform weight.

* from fbprophet import Prophet: is Facebook prophet which required requires a pandas DataFrame with two columns:
---
- ds, for datestamp, is a datestamp column in a format expected by pandas.
- y, a numeric column containing the measurement we wish to forecast.

* import pmdarima as pm: (originally pyramid-arima, for the anagram of 'py' + 'arima') is a statistical library designed to fill the void in Python's time series analysis capabilities. This includes: The equivalent of R's auto.arima functionality. A collection of statistical tests of stationarity and seasonality. Time series utilities, such as differencing and inverse differencing.

## Dashboard

An Excel dashboard is a place where we are going to track all our important indicators, metrics, and data points
using visuals and charts.

This Dashboard will giveus a high-level view of how this 5 business are doing and help us make snap decisions based on data.

Here we are presenting an Analytical Dashboard that will help us save time in the analysis and presentation of our data











