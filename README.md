# Final_Project

### Selected Topic:
- Our selected topic entails using machine learning to predict future stock prices based on historical data. The machine learning model will take in stock data and will from the last three years and will output prediction prices of the S&P 500. Our dataset conatins the monthly stock price of the S&P 500 from the start of 2019.

### Reason we selected the topic:
- We selected this topic due to our interest in the stock market and desire to create a machine learning model that could potentially predict market trends based on previous performance data. 

- We wanted to predict something relevant to the our economy. After analyzing various data sets, we decided we wanted to better understand the S&P 500. We each picked an individual sector with a stock in the   S&P 500 of interest to learn more about.

### Background Info on S&P 500:
- The S&P 500 Index, or Standard & Poor's 500 Index, is a market-capitalization-weighted index of 500 leading publicly traded companies in the U.S. It is not an exact list of the top 500 U.S. companies by market cap because there are other criteria that the index includes. Still, the S&P 500 index is regarded as one of the best gauges of prominent American equities' performance and the stock market overall.

### Description of the source of data:
- Daily starting and ending price of the S&P 500 from 2019.

- Daily prices for our handpicked stocks from 2019:
    (Apple, Nike, Kellogg, Occidental Petroleum, and C.H. Robinson)

- Daily prices for their individual sectors: 
(Tech, Consumer Discretionary, Consumer Staples, Energy, Industrials)


### Questions we hope to answer with the data:
- We hope to find out if it is possible to create a machine learning model that can accurately predict stock prices. Specifically, we want to find out if we can accurately predict the S&P 500 over a certain period of time. 

Question : How accurately can we predict stock prices based on our created machine learning models?

- Another question we want to answer is if there is a relationship between historical trends and future stock performance.

Question: Is there a relationship between historical trends and future stock performance?

### Communication Protocls
- Our group outlined some standard communication protocols. For any infromation relvant to the whole group, we deliberate through our group slack channel. If there is smiething specific between two members, then the members reach out to each other through slcak via direct message.

### Outline
- Introduction to our topic
    - Overview of S&P 500 and our selected stocks
    - Basics of S&P 500
    - Our hand picked stocks
- Reason why we selected our topic 
- Description of our source of data 
- Questions they hope to answer with the data 
- Description of Analysis
    - Description of the data exploration phase of the project
    - Description of the analysis phase of the project 
- Technologies, languages, tools, and algorithms used throughout the project 
- Result of analysis
    - Recommendation for future analysis 
    - Anything the team would have done differently
- Takeaways and Conclusion


### Database Rubric Questions
It's easy to view where each of these components in more depth as they are highlighted in our code located in the ![Database_Management](/Database_Management/) folder

- Database stores static data for use during the project
Database stores static data that is taken at the end of the trading day, as opposed to streaming live updated data.

- Database interfaces with the project in some format (e.g., scraping updates the database, or database connects to the model)
Database will connect directly to the model, using a python code in Jupyter notebook.

- Includes at least two tables (or collections, if using MongoDB)
There are 11 tables total in the Database, that contain daily stock data between Jan 2019 and Dec 2021.

- Includes at least one join using the database language (not including any joins in Pandas)
Each stock table is joined to a Sector table for output into the model.  Each of these joined tables is also joined again with the S&P table.


![ERD](/Database_Management/KU_QuickDBD_ERD.png) 


- The ERD with relationships is provided for the 5 stock tables and related S&P table. It can also be viewed in the Database_Management folder 

- Includes one connection string with SQL and Machine Learning Models. We used postgreSQL for the SQL interface.

### Machine Learning Questions
More in depth analysis regarding each of these questions can be viewed in the presentation and ![Machine_Learning](/Machine_Learning_Final/) folder and other Machine_Learning folders.

- Description of data preprocessing: The data was pulled in from Postgres sql to be read into a pandas data frame

- Description of feature engineering and the feature selection, including the team's decision-making process: The team decided on three predictive modeling techniques based on how well known and respected each model was    

- Description of how data was split into training and testing sets: The data was split into training and testing sets with the first 80% of the data in the training set and the last 20% in the testing set. This was done in order of dates and could not be random due to time series limitations.     

- Description of how model was trained: The model was trained by using the training data and getting a sense for the seasonality. Then after looking at graphs it was decided to perform a log transformation on the data to get a better fit for the outcome.     

- Description and explanation of model’s accuracy: The models performed fairly and displayed an average prediction but the models were not able to follow seasonality as we would have liked. Still, it was fairly accurate and our Prophet model stood out with the smallest error. The model does predict stocks and can say that when the S&P goes up how the corresponding sector will react.

- Machine Learning Models
Linear Regression
The purpose of viewing the linear regression model was to see how closely correlated each company’s stock or industry was related to the S&P (i.e. if the S&P increased would the company or industry also increase).

Arima Model
ARIMA: Autoregressive Integrated Moving Average. This is a statistical model that attempts to use past observations of the target variable to forecast its future values. Some limitations to the ARIMA model is that it has difficulty predicting turning points, it struggles with seasonality, and it performs well on short term forecasts but has poorer performance long term. This model was the second best and did a good job predicting an estimated average of where the stock prices would go. This model did not perform as well when it comes to seasonality/trend.

Facebook Prophet Model
The Facebook Prophet model is an additive regression model (like the ARIMA) but it includes growth trend and seasonal components. Some limitations of the Prophet model are it has a tendency to overfit the data, and it requires data to be in a specific format. Overall this model performed the best and was more accurate in terms on seasonality and trend. 


### Link to our Website
https://roark23.github.io/Final_Project/Website/overview_report.html


### Presentation Slides
- Here is the link to our Google Slides Presentation:
https://docs.google.com/presentation/d/1LKO1Rd-QlpozrJrnYhYHzJuyGJvUBEsal7sHNvLBlyc/edit#slide=id.g134b22ff32b_0_37

### Note all another descrptive information required for this deliverabe can be viewed in their respective markdowns.