# Final_Project


# Technologies Used

* Git Hub:
   - Is a provider of Internet hosting for software development and version control using Git. It offers the distributed version control and source code management (SCM) functionality of Git, plus its own features. It provides access control and several collaboration features.

* Visual Studio Code:
   - Is a source-code editor that we used with a variety of programming languages, including Python. It is based on the Electron framework,    which   is used to develop  Web applications that run on the Blink layout engine.

* Kaggle:
   - Is an online community of data scientists and machine learning practitioners. Kaggle allowed us to find our data sets for this project.

* Phyton 3:
   - functions are reusable, ordered chunk of code that performs the single, connected activity. Functions provide our program with more modularity and allow us to reuse a lot of code.

* SQL Database:
   - SQL, is short for Structured Query Language, is a database programming language for querying and editing information.

* Quick DBD - www.quickdatabasediagrams.com: 
   - Starting the build of the database, we are using the same ERD program used in class, the Quick DBD.

* Tableau:
   - Is an American interactive data visualization software that we are using to present our DASHBOARD.

* Snipping Tool:
   - Is a screenshot utility included in windows Vista and later versions that can take screenshots in a variety of ways, which we use to create the storyboard for this project.

* Excel:
   - Is a spreadsheet program from Microsoft and a component of its Office product group for business applications.

* JSON File:
   -  Is a file that stores simple data structures and objects in JavaScript Object Notation (JSON) format, which is a standard data interchange format. It is primarily used for transmitting data between a web application and a server.

* index.html:
   - As its name suggests index.html consists of two words index and html. The index simply means shortlist for multiple items. In web terminology generally used to showcase the website pages, categories or parts on a single page.

* HTML: 
   - Hypertext Markup Language, a standardized system for tagging text files to achieve font, color, graphic, and hyperlink effects on World Wide Web pages.

* Heroku:
   - Is a Platform as a Service  (Paas) cloud offering that provisions the development of Web applications and services. It is a cloud application development platform that provides development tools, scalable processing power and cloud management functionality.

---

## Data Cleaning and Analysis

Data cleansing involves spotting and resolving potential data inconsistencies or errors to improve our data quality.
In this process, we review, analyze, detect, modify, or remove “dirty” data to make our dataset “clean.” Data clensing 
is also called data cleaning or data scrubbing.

The way we check for unique values is to use the Pandas DataFrame's .

Pandas: will be used to clean the data and perform an exploratory analysis. Further analysis is completed using Python.

## Database Storage

Selecting the right data store for our requirements is a key design decision. Reason why we have decided to use
PostgresSQL Database as an standard language for accessing and manipulating databases, which provides various benefits to our development, 
such as saving time by automating routine tasks, locating and fixing errors, taking advantage of intelligent support from 
the IDE, and increasing overall productivity.

yfinance: is used to download and display information on our selected stocks.

-- creating tables for stock index
CREATE TABLE spstock (
    date DATE NOT NULL,
    opening FLOAT NOT NULL,
    low FLOAT NOT NULL,
    cloasing FLOAT NOT NULL,
    adj_closing FLOAT NOT NULL,
    volume DECIMAL NOT NULL,
    PRIMARY KEY (date),
    UNIQUE (date)
);

-- DROP TABLE spstock;
SELECT * FROM spstock;

## Machine Learning

Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that we learn, and gradually improving its accuracy. Through the use of statistical methods, algorithms are trained to make classifications or predictions, uncovering key insights within our data mining project. Also, The Evaluation metric measures the quality of the machine learning model. ideally impacting key growth metrics.

The machine learning aspect will take in stock data from the last three years of the dataset and will output prediction prices. The code will run two different forecasting models (ARIMA and Facebook Prophet) and will produce graphs displaying the predictions to compare each of the models.

* import numpy as np: keeps our code more readable.

* import pandas as pd: tells Phyton to bring the pandas data analysis library into our current environment.

* import matplotlib.pyplot as plt: is importer into our namespace under the shorter name plt. The pyplot is where the plot () scatter(), and other commands live

* from pandas.plotting import lag_plot: is a scatter plot for a time series and the same data lagged.

* import datetime as dt: Return a string stating the day of the week corresponding to datetime dt .

* from sklearn.linear_model import LinearRegression: Is one of the best statistical models that studies the relationship between a dependent variable (Y) with a given set of independent variables (X). The relationship is established with the help of fitting a best line. sklearn.linear_model.LinearRegression is the module used to implement linear regression.

* from sklearn.model_selection import train_test_split: is a function in Sklearn model selection for splitting data arrays into two subsets: for training data and for testing data. With this function, we do not need to divide the dataset manually. By default, Sklearn train_test_split will make random partitions for the two subsets.

* from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error: Array-like value defines weights used to average errors. ‘raw_values’ : Returns a full set of errors in case of multioutput input. ‘uniform_average’ : Errors of all outputs are averaged with uniform weight.

* from fbprophet import Prophet: is Facebook prophet which required requires a pandas DataFrame with two columns:

    - ds, for datestamp, is a datestamp column in a format expected by pandas.
    - y, a numeric column containing the measurement we wish to forecast.

* import pmdarima as pm: (originally pyramid-arima, for the anagram of 'py' + 'arima') is a statistical library designed to fill the void in Python's time series analysis capabilities. This includes: The equivalent of R's auto.arima functionality. A collection of statistical tests of stationarity and seasonality. Time series utilities, such as differencing and inverse differencing.

## Dashboard

Here we are presenting an Analytical Dashboard that will help us save time in the analysis and presentation of our data.

This Dashboard will give us a high-level view of how this business are doing and help us make snap decisions based on data.

An Excel file will be used and transformed from .csv, we are going to track all our important indicators, metrics, and data points using visuals and charts.

To be presented.















