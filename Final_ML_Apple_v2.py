#!/usr/bin/env python
# coding: utf-8

# Create a connection to database

# In[1]:


import numpy as np 
import pandas as pd
import plotly
import plotly.figure_factory as ff
import plotly.graph_objects as go
from datetime import datetime
import plotly.express as px
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
import datetime as dt
import itertools
from IPython.display import HTML
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from fbprophet import Prophet
import math


# In[2]:


# Install psycopg2 module
# pip install psycopg2

# Import dependencies
import psycopg2
import pandas.io.sql as psql


# In[4]:


# Database credentials
conn = psycopg2.connect(
    host = "localhost",
    database = "BootCamp_Final",
    user = "postgres",
    password = "Ku345226$")

# Use curser method to execute statements
cur = conn.cursor()

# Extract and Load data into a dataFrame
tech_sector_pd = psql.read_sql('SELECT * FROM tech_sector_export', conn)
print(tech_sector_pd)

# Close curser
cur.close()

# Close connection
conn.close()


# In[5]:


#tech_sector_pd = pd.read_csv('KU_tech_sector_export.csv')
tech_sector_pd['date'] = pd.to_datetime(tech_sector_pd['date'])
tech_sector_pd = tech_sector_pd.set_index('date')


# In[6]:


df_apple = tech_sector_pd[['apple']]


# In[7]:


#plot
apple_closing_figure = go.Figure(go.Scatter(x=df_apple.index, y=df_apple['apple']))
apple_closing_figure.update_layout(title='Apple Closing Price')
apple_closing_figure.update_yaxes(type='linear')


# In[8]:


plt.figure(figsize=(16,8))
sp_closing_figure = go.Figure(go.Scatter(x=tech_sector_pd.index, y=tech_sector_pd['spstock']))
sp_closing_figure.update_layout(title='S&P Closing Price')
sp_closing_figure.update_yaxes(type='linear')


# In[9]:


plt.figure(figsize=(16,8))
tech_closing_figure = go.Figure(go.Scatter(x=tech_sector_pd.index, y=tech_sector_pd['tech_sector']))
tech_closing_figure.update_layout(title='Tech Closing')
tech_closing_figure.update_yaxes(type='linear')


# In[10]:


df_apple_close = df_apple[['apple']]
apple_list = np.array(df_apple['apple'], dtype=float)
hist_data = [apple_list]
group_labels=['Apple']
apple_dist = ff.create_distplot(hist_data, group_labels)
apple_dist.show()


# In[11]:


df_sp_close = tech_sector_pd[['spstock']]
sp_list = np.array(tech_sector_pd['spstock'], dtype=float)
hist_data = [sp_list]
group_labels=['S&P']
sp_dist = ff.create_distplot(hist_data, group_labels)
sp_dist.show()


# In[12]:


df_tech_close = tech_sector_pd[['tech_sector']]
tech_list = np.array(tech_sector_pd['tech_sector'], dtype=float)
hist_data = [tech_list]
group_labels=['Tech']
tech_dist = ff.create_distplot(hist_data, group_labels)
tech_dist.show()


# In[13]:


def plot_seasonal_decompose(result:DecomposeResult, dates:pd.Series=None, title:str="Seasonal Decomposition"):
    x_values = dates if dates is not None else np.arange(len(result.observed))
    return (
        make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"],
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.observed, mode="lines", name='Observed'),
            row=1,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.trend, mode="lines", name='Trend'),
            row=2,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.seasonal, mode="lines", name='Seasonal'),
            row=3,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.resid, mode="lines", name='Residual'),
            row=4,
            col=1,
        )
        .update_layout(
            height=900, title=f'<b>{title}</b>', margin={'t':100}, title_x=0.5, showlegend=False
        )
    )


# In[14]:


result = seasonal_decompose(apple_list, model='multiplicative', freq=30)
apple_seasonal = plot_seasonal_decompose(result)
apple_seasonal.update_layout(title='Apple Seasonal Decomposition')


# In[15]:


#Looking at trend and seasonality from time series
result = seasonal_decompose(tech_list, model='multiplicative', freq=30)
tech_seasonal = plot_seasonal_decompose(result)
tech_seasonal.update_layout(title='Tech Seasonal Decomposition')


# Since data is not stationary we will preform log transformation to eliminate trend

# In[16]:


apple_log = np.log(apple_list)
df_apple_log = pd.DataFrame(apple_log, columns=['apple'])
df_apple_log['date'] = df_apple_close.index


# In[17]:


tech_log = np.log(tech_list)
df_tech_log = pd.DataFrame(tech_log, columns=['tech_sector'])
df_tech_log['date'] = df_tech_close.index


# Split data into train and test sets

# In[18]:


apple_train_data= pd.DataFrame(df_apple_log.iloc[:int(df_apple_log.shape[0]*0.8)])
apple_test_data = pd.DataFrame(df_apple_log.iloc[int(df_apple_log.shape[0]*0.80):])
apple_test_train_fig = go.Figure()
apple_test_train_fig.add_trace(go.Scatter(x=apple_train_data['date'], y=apple_train_data['apple'], name='Train'))
apple_test_train_fig.add_trace(go.Scatter(x=apple_test_data['date'], y=apple_test_data['apple'], name='Test'))
apple_test_train_fig.update_layout(title='Apple Test Train Data')


# In[19]:


tech_train_data= df_tech_log.iloc[:int(df_tech_log.shape[0]*0.8)]
tech_test_data = df_tech_log.iloc[int(df_tech_log.shape[0]*0.80):]
tech_test_train_fig = go.Figure()
tech_test_train_fig.add_trace(go.Scatter(x=tech_train_data['date'], y=tech_train_data['tech_sector'], name='Train'))
tech_test_train_fig.add_trace(go.Scatter(x=tech_test_data['date'], y=tech_test_data['tech_sector'], name='Test'))
tech_test_train_fig.update_layout(title='Tech Test Train Data')


# This Machine Learning Model will run a linear regression, ARIMA, and a Facebook Prophet Model -some limitations of these models will be they do not take into accoutn current world sitations (like COVID impacts on the economy)
# 
# Linear Regression Model

# In[20]:


df_linear_apple = tech_sector_pd[['apple', 'spstock']]


# In[21]:


df_linear_tech = tech_sector_pd[['tech_sector', 'spstock']]


# In[22]:


# for linear regression model we need an x_train value, and a y_train value
X_train, X_test, y_train, y_test = train_test_split(df_linear_apple[['apple']], df_linear_apple[['spstock']], test_size=.2)

linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)

y_pred = pd.DataFrame(linear_regression_model.predict(X_test))

apple_linear_mse = mean_squared_error(y_test, y_pred)
apple_linear_mae = mean_absolute_error(y_test, y_pred)
apple_linear_rmse = math.sqrt(mean_squared_error(y_test, y_pred))


# In[23]:


apple_linear_regression = go.Figure()
apple_linear_regression.add_trace(go.Scatter(x=X_train['apple'], y=y_train['spstock'], mode='markers', name='Train Data'))
apple_linear_regression.add_trace(go.Scatter(x=X_test['apple'], y=y_pred[0], name='Prediction'))
apple_linear_regression.update_xaxes(type='linear')
apple_linear_regression.update_yaxes(type='linear')
apple_linear_regression.update_layout(title='Apple vs S&P Linear Regression')


# In[24]:


# for linear regression model we need an x_train value, and a y_train value
X_train, X_test, y_train, y_test = train_test_split(df_linear_tech[['tech_sector']], df_linear_apple[['spstock']], test_size=.2)

linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)

y_pred = pd.DataFrame(linear_regression_model.predict(X_test))

tech_linear_mse = mean_squared_error(y_test, y_pred)
tech_linear_mae = mean_absolute_error(y_test, y_pred)
tech_linear_rmse = math.sqrt(mean_squared_error(y_test, y_pred))


# In[25]:


tech_linear_regression = go.Figure()
tech_linear_regression.add_trace(go.Scatter(x=X_train['tech_sector'], y=y_train['spstock'], mode='markers', name='Train Data'))
tech_linear_regression.add_trace(go.Scatter(x=X_test['tech_sector'], y=y_pred[0], name='Prediction'))
tech_linear_regression.update_xaxes(type='linear')
tech_linear_regression.update_yaxes(type='linear')
tech_linear_regression.update_layout(title='Tech Sector vs S&P Linear Regression')


# ARIMA Model
# 
# For Apple

# In[26]:


#Modeling
arima_model = ARIMA(apple_train_data['apple'], order=(1,1,0))
arima_fitted = arima_model.fit()


# RUNNING THE L-BFGS-B CODE
# 
#            * * *
# 
# Machine precision = 2.220D-16
#  N =            2     M =           12
# 
# At X0         0 variables are exactly at the bounds
# 
# At iterate    0    f= -2.37060D+00    |proj g|=  3.32236D-01
# 
# At iterate    5    f= -2.37062D+00    |proj g|=  7.00837D-02
# 
# At iterate   10    f= -2.37065D+00    |proj g|=  2.12351D-02
# 
#            * * *
# 
# Tit   = total number of iterations
# Tnf   = total number of function evaluations
# Tnint = total number of segments explored during Cauchy searches
# Skip  = number of BFGS updates skipped
# Nact  = number of active bounds at final generalized Cauchy point
# Projg = norm of the final projected gradient
# F     = final function value
# 
#            * * *
# 
#    N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
#     2     14     31      1     0     0   1.279D-05  -2.371D+00
#   F =  -2.3706456645958842     
# 
# CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             

# In[27]:


#Forecast
arima_forecast, se, conf = arima_fitted.forecast(len(apple_test_data), alpha=0.05)

arima_fc_series = pd.DataFrame(arima_forecast, index=apple_test_data['date'])
lower_series = pd.DataFrame(conf[:,0], index=apple_test_data['date'])
upper_series = pd.DataFrame(conf[:,1], index=apple_test_data['date'])


# In[28]:


#ARIMA Plot
apple_arima = go.Figure()
apple_arima.add_trace(go.Scatter(x=apple_train_data['date'], y=apple_train_data['apple'], name='Train'))
apple_arima.add_trace(go.Scatter(x=apple_test_data['date'], y=apple_test_data['apple'], name='Test'))
apple_arima.add_trace(go.Scatter(x=arima_fc_series.index, y=arima_fc_series[0], name='Forecast'))
apple_arima.add_trace(go.Scatter(x=lower_series.index, y=lower_series[0], fill='tonexty', fillcolor='rgba(0,100,80,0.1)', name='Lower Bound'))
apple_arima.add_trace(go.Scatter(x=upper_series.index, y=upper_series[0], fill='tonexty', fillcolor='rgba(0,100,80,0.1)', name='Upper Bound'))
apple_arima.update_layout(title='Apple ARIMA')


# In[29]:


# ARIMA Model Statistics
apple_arima_mse = mean_squared_error(apple_test_data['apple'], arima_forecast)
apple_arima_mae = mean_absolute_error(apple_test_data['apple'], arima_forecast)
apple_arima_rmse = math.sqrt(mean_squared_error(apple_test_data['apple'], arima_forecast))


# ARIMA Model for Tech Sector

# In[30]:


#Modeling
arima_model = ARIMA(tech_train_data['tech_sector'], order=(2,1,2))
arima_fitted = arima_model.fit()


# RUNNING THE L-BFGS-B CODE
# 
#            * * *
# 
# Machine precision = 2.220D-16
#  N =            5     M =           12
# 
# At X0         0 variables are exactly at the bounds
# 
# At iterate    0    f= -2.64170D+00    |proj g|=  7.98821D-02
# 
# At iterate    5    f= -2.64170D+00    |proj g|=  5.73848D-02
# 
# At iterate   10    f= -2.64176D+00    |proj g|=  6.28197D-01
# 
# At iterate   15    f= -2.64194D+00    |proj g|=  8.46700D-03
# 
# At iterate   20    f= -2.64242D+00    |proj g|=  1.46989D-01
# 
# At iterate   25    f= -2.64281D+00    |proj g|=  4.83080D-03
# 
# At iterate   30    f= -2.64282D+00    |proj g|=  8.77540D-02
# 
# At iterate   35    f= -2.64283D+00    |proj g|=  4.75966D-03
# 
# At iterate   40    f= -2.64283D+00    |proj g|=  6.38130D-03
# 
# At iterate   45    f= -2.64284D+00    |proj g|=  1.82605D-03
# 
# At iterate   50    f= -2.64284D+00    |proj g|=  1.52767D-05
# 
#            * * *
# 
# Tit   = total number of iterations
# Tnf   = total number of function evaluations
# Tnint = total number of segments explored during Cauchy searches
# Skip  = number of BFGS updates skipped
# Nact  = number of active bounds at final generalized Cauchy point
# Projg = norm of the final projected gradient
# F     = final function value
# 
#            * * *
# 
#    N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
#     5     50     62      1     0     0   1.528D-05  -2.643D+00
#   F =  -2.6428367566636681     
# 
# CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             

# In[31]:


#Forecast
arima_forecast, se, conf = arima_fitted.forecast(len(tech_test_data), alpha=0.05)

arima_fc_series = pd.DataFrame(arima_forecast, index=tech_test_data['date'])
lower_series = pd.DataFrame(conf[:,0], index=tech_test_data['date'])
upper_series = pd.DataFrame(conf[:,1], index=tech_test_data['date'])


# In[32]:


#ARIMA Plot
tech_arima = go.Figure()
tech_arima.add_trace(go.Scatter(x=tech_train_data['date'], y=tech_train_data['tech_sector'], name='Train'))
tech_arima.add_trace(go.Scatter(x=tech_test_data['date'], y=tech_test_data['tech_sector'], name='Test'))
tech_arima.add_trace(go.Scatter(x=arima_fc_series.index, y=arima_fc_series[0], name='Forecast'))
tech_arima.add_trace(go.Scatter(x=lower_series.index, y=lower_series[0], fill='tonexty', fillcolor='rgba(0,100,80,0.1)', name='Lower Bound'))
tech_arima.add_trace(go.Scatter(x=upper_series.index, y=upper_series[0], fill='tonexty', fillcolor='rgba(0,100,80,0.1)', name='Upper Bound'))
tech_arima.update_layout(title='Tech ARIMA')


# In[33]:


# ARIMA Model Statistics
tech_arima_mse = mean_squared_error(tech_test_data['tech_sector'], arima_forecast)
tech_arima_mae = mean_absolute_error(tech_test_data['tech_sector'], arima_forecast)
tech_arima_rmse = math.sqrt(mean_squared_error(tech_test_data['tech_sector'], arima_forecast))


# Facebook Prophet Model for Apple

# In[20]:


prophet_data = pd.DataFrame()
prophet_data['y'] = df_apple_log['apple']
prophet_data['ds'] = tech_sector_pd.index


#train and validation
prophet_train = prophet_data[:int(prophet_data.shape[0]*0.80)]
prophet_test = prophet_data[int(prophet_data.shape[0]*0.80):]

#fit the model
prophet_model = Prophet(interval_width=0.95)
prophet_model.fit(prophet_train)

#predictions
close_prices = prophet_model.make_future_dataframe(periods=212)
forecast = prophet_model.predict(close_prices)


# Initial log joint probability = -2.23024
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#       99       2572.78   0.000747485       620.524           1           1      118   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#      199        2595.3   0.000294246       1826.88      0.6393      0.6393      226   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#      299        2642.5    0.00326795       1808.16       4.602      0.4602      335   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#      399       2665.82   0.000970188       540.408           1           1      453   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#      499       2669.96   0.000345003       1391.64      0.3282      0.3282      564   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#      599       2675.76   0.000290243       270.403      0.7861      0.7861      670   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#      699        2681.7   0.000843919       291.271           1           1      780   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#      799       2687.45    0.00020713       428.196           1           1      891   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#      899       2690.79   9.46654e-05       228.718           1           1     1011   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#      975       2691.54   2.30685e-05       422.248   1.404e-07       0.001     1143  LS failed, Hessian reset 
#      999       2692.15   0.000611202       290.213      0.3399           1     1170   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     1062       2692.86   8.52796e-06       145.561   8.558e-08       0.001     1334  LS failed, Hessian reset 
#     1086       2693.03   9.74659e-06        168.08   1.035e-07       0.001     1412  LS failed, Hessian reset 
#     1099       2693.08    7.4496e-05       262.406      0.4762      0.4762     1427   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     1199       2693.82    0.00320576       491.157           1           1     1553   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     1299       2695.87   0.000766041       299.391           1           1     1672   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     1303       2695.88   9.55353e-06       166.184   1.077e-07       0.001     1728  LS failed, Hessian reset 
#     1372        2696.2   6.17934e-06       169.252   4.029e-08       0.001     1862  LS failed, Hessian reset 
#     1399       2696.32   0.000572979       123.637           1           1     1891   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     1453       2696.53    7.6508e-06       214.123   3.047e-08       0.001     2004  LS failed, Hessian reset 
#     1483       2696.63   3.29792e-06       96.1223    2.61e-08       0.001     2086  LS failed, Hessian reset 
#     1499       2696.64   3.56021e-05       200.197      0.2525      0.8544     2105   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     1599       2696.86   2.34227e-06       78.2492      0.2823      0.2823     2239   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     1699       2697.31    0.00220044       215.136      0.2888           1     2360   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     1706       2697.35   6.77447e-06       191.434   2.158e-08       0.001     2418  LS failed, Hessian reset 
#     1799       2697.63    5.5121e-05       136.389      0.2549           1     2541   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     1800       2697.63   3.30824e-06       95.5144   2.426e-08       0.001     2584  LS failed, Hessian reset 
#     1899       2697.71   5.57669e-06       71.2698      0.3474      0.3474     2708   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     1999       2698.46    0.00024277        298.46      0.3974      0.3974     2821   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     2043       2698.61   3.26628e-06       99.8382   3.764e-08       0.001     2924  LS failed, Hessian reset 
#     2099       2698.67   3.84805e-05        79.867           1           1     2998   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     2199       2698.89   0.000557568       181.368           1           1     3116   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     2202       2698.89   5.30043e-06        145.65   3.664e-08       0.001     3163  LS failed, Hessian reset 
#     2299       2698.99   2.43249e-05       113.534      0.2136      0.7237     3287   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     2399       2699.04   9.56569e-07       56.6122      0.6835      0.6835     3410   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     2411       2699.04   6.47587e-06       164.502    5.49e-08       0.001     3463  LS failed, Hessian reset 
#     2499       2699.11   0.000280527       133.776      0.2434           1     3568   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     2538       2699.44   6.94433e-05       576.657   1.125e-07       0.001     3656  LS failed, Hessian reset 
#     2579       2699.83   2.95785e-06       96.0807   3.392e-08       0.001     3751  LS failed, Hessian reset 
#     2599       2699.96   0.000601671       555.995           1           1     3772   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     2644       2700.08   3.66131e-06       113.275   3.197e-08       0.001     3878  LS failed, Hessian reset 
#     2699       2700.12   7.93864e-07       54.2047      0.3007      0.3007     3961   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     2712       2700.12   4.17139e-07        53.586           1           1     3980   
# Optimization terminated normally: 
#   Convergence detected: relative gradient magnitude is below tolerance

# In[21]:


forecast = forecast.set_index('ds')
forecast = forecast.rename(columns={'yhat':'Prediction'})
forecast = forecast[forecast.index.dayofweek < 5]
forecast = forecast.loc['2021-05-27':'2021-12-31']


# In[6]:


#plot
apple_prophet = go.Figure()
apple_prophet.add_trace(go.Scatter(x=apple_train_data['date'], y=apple_train_data['apple'], name='Train'))
apple_prophet.add_trace(go.Scatter(x=apple_test_data['date'], y=apple_test_data['apple'], name='Test'))
apple_prophet.add_trace(go.Scatter(x=forecast.index, y=forecast['Prediction'], name='Forecast'))
apple_prophet.add_trace(go.Scatter(x=forecast.index, y=forecast['yhat_lower'], fill='tonexty', fillcolor='rgba(0, 100, 80, 0.1)', name='Lower Bound'))
apple_prophet.add_trace(go.Scatter(x=forecast.index, y=forecast['yhat_upper'], fill='tonexty', fillcolor='rgba(0, 100, 80, 0.1)', name='Upper Bound'))
apple_prophet.update_layout(title='Apple Prophet')


# In[7]:


#Prophet Model Statistics
apple_prophet_mse = mean_squared_error(apple_test_data['apple'], forecast['Prediction'])
apple_prophet_mae = mean_absolute_error(apple_test_data['apple'], forecast['Prediction'])
apple_prophet_rmse = math.sqrt(mean_squared_error(apple_test_data['apple'], forecast['Prediction']))


# In[8]:


prophet_data = pd.DataFrame()
prophet_data['y'] = df_tech_log['tech_sector']
prophet_data['ds'] = tech_sector_pd.index

#train and validation
prophet_train = prophet_data[:int(prophet_data.shape[0]*0.80)]
prophet_test = prophet_data[int(prophet_data.shape[0]*0.80):]

#fit the model
prophet_model = Prophet(interval_width=0.95)
prophet_model.fit(prophet_train)

#predictions
close_prices = prophet_model.make_future_dataframe(periods=212)
forecast = prophet_model.predict(close_prices)


# Initial log joint probability = -2.0755
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#       99       2694.34     0.0157375       2557.43           1           1      125   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#      199       2770.86    0.00103004       991.445           1           1      236   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#      299        2807.2    0.00184741       1876.19           1           1      343   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#      399       2844.91    0.00220445       990.696          10           1      453   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#      499       2854.71     0.0391582       3474.16           1           1      563   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#      599       2877.07    0.00226096       415.378           1           1      680   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#      699       2880.22      0.001232       815.993           1           1      792   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#      799       2888.69    0.00198285       1919.41           1           1      896   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#      899       2890.86    0.00668013       1464.32           1           1     1010   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#      994       2891.73   2.69432e-06       91.5758   9.791e-09       0.001     1219  LS failed, Hessian reset 
#      999       2891.74   9.56993e-05       310.416           1           1     1224   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     1099       2893.29   0.000264708       214.655      0.4995      0.4995     1339   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     1199       2893.96   0.000170513        583.66       0.457       0.457     1452   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     1210          2894   4.95545e-06       232.366   1.101e-08       0.001     1518  LS failed, Hessian reset 
#     1275       2894.54   1.06005e-05         245.5   1.172e-07       0.001     1652  LS failed, Hessian reset 
#     1299       2894.58   7.07185e-05       443.648      0.0777           1     1679   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     1399       2895.11   0.000493772        232.82           1           1     1804   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     1499       2896.45      0.012811       588.903           1           1     1919   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     1599       2897.58   0.000285926       230.192           1           1     2035   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     1620       2897.71   2.37123e-06       121.041   3.208e-08       0.001     2112  LS failed, Hessian reset 
#     1699       2897.95   2.14951e-05       96.8394      0.4993      0.4993     2209   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     1799       2898.44    0.00033028       991.558           1           1     2327   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     1899       2898.75    0.00359775       1828.42           1           1     2446   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     1999       2899.34   7.78992e-05       287.362           1           1     2561   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     2008       2899.37   2.63967e-06       130.598   1.197e-08       0.001     2615  LS failed, Hessian reset 
#     2099       2899.48    0.00116376       322.312      0.3341      0.3341     2718   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     2199       2900.29   9.71092e-05        456.13      0.8907      0.8907     2849   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     2227       2900.34   6.32593e-06       208.342   8.022e-08       0.001     2924  LS failed, Hessian reset 
#     2262        2900.4   9.36625e-06       297.951   3.095e-08       0.001     3003  LS failed, Hessian reset 
#     2284       2900.44    3.4464e-06       135.736   5.352e-08       0.001     3084  LS failed, Hessian reset 
#     2299       2900.45   1.14592e-05       77.1191      0.4765      0.4765     3101   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     2354       2900.49   2.34357e-06       122.717   1.246e-08       0.001     3218  LS failed, Hessian reset 
#     2392       2900.53   6.25607e-06       243.798   4.612e-08       0.001     3315  LS failed, Hessian reset 
#     2399       2900.54   5.53982e-06       184.952      0.1327      0.1327     3323   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     2499       2900.69    0.00406764       2114.42           1           1     3455   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     2599       2903.76    0.00179323        419.55           1           1     3567   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     2649       2904.15   7.28623e-06       394.791   1.405e-08       0.001     3672  LS failed, Hessian reset 
#     2699       2904.33   0.000237563       170.196           1           1     3728   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     2799       2904.58    0.00249449       1301.38           1           1     3846   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     2812       2904.66   3.30771e-06       156.358   9.891e-09       0.001     3912  LS failed, Hessian reset 
#     2899        2904.9   8.15293e-05       662.047      0.5543      0.5543     4018   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     2969       2904.99   3.51759e-06       166.237    1.05e-08       0.001     4147  LS failed, Hessian reset 
#     2999       2904.99   0.000141372       55.2093      0.4445           1     4184   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     3099       2905.09   8.54488e-06       37.8129      0.3874           1     4322   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     3171       2905.13   4.83347e-06       272.468   1.345e-08       0.001     4474  LS failed, Hessian reset 
#     3199       2905.18   3.34266e-05       142.674       0.344           1     4506   
#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     3204       2905.18   2.77713e-06       45.2101   5.652e-08       0.001     4555  LS failed, Hessian reset 
#     3218       2905.18   2.63228e-07       32.4741   6.896e-09       0.001     4618  LS failed, Hessian reset 
# Optimization terminated normally: 
#   Convergence detected: relative gradient magnitude is below tolerance

# In[9]:


forecast = forecast.set_index('ds')
forecast = forecast.rename(columns={'yhat':'Prediction'})
forecast = forecast[forecast.index.dayofweek < 5]
forecast = forecast.loc['2021-05-27':'2021-12-31']


# In[10]:


#plot
tech_prophet = go.Figure()
tech_prophet.add_trace(go.Scatter(x=tech_train_data['date'], y=tech_train_data['tech_sector'], name='Train'))
tech_prophet.add_trace(go.Scatter(x=tech_test_data['date'], y=tech_test_data['tech_sector'], name='Test'))
tech_prophet.add_trace(go.Scatter(x=forecast.index, y=forecast['Prediction'], name='Forecast'))
tech_prophet.add_trace(go.Scatter(x=forecast.index, y=forecast['yhat_lower'], fill='tonexty', fillcolor='rgba(0, 100, 80, 0.1)', name='Lower Bound'))
tech_prophet.add_trace(go.Scatter(x=forecast.index, y=forecast['yhat_upper'], fill='tonexty', fillcolor='rgba(0, 100, 80, 0.1)', name='Upper Bound'))
tech_prophet.update_layout(title='Tech Prophet')


# In[11]:


#Prophet Model Statistics
tech_prophet_mse = mean_squared_error(tech_test_data['tech_sector'], forecast['Prediction'])
tech_prophet_mae = mean_absolute_error(tech_test_data['tech_sector'], forecast['Prediction'])
tech_prophet_rmse = math.sqrt(mean_squared_error(tech_test_data['tech_sector'], forecast['Prediction']))


# In[12]:


cell_hover = {  # for row hover use <tr> instead of <td>
    'selector': 'td:hover',
    'props': [('background-color', '#ffffb3')]
}
index_names = {
    'selector': '.index_name',
    'props': 'font-style: italic; color: darkgrey; font-weight:normal;'
}
headers = {
    'selector': 'th:not(.index_name)',
    'props': 'background-color: #000066; color: white;'
}


# In[13]:


tech_statistics = pd.DataFrame(index=['MSE', 'MAE','RMSE'])
tech_statistics['Linear'] = [tech_linear_mse, tech_linear_mae, tech_linear_rmse]
tech_statistics['ARIMA'] = [tech_arima_mse, tech_arima_mae, tech_arima_rmse]
tech_statistics['Prophet'] = [tech_prophet_mse, tech_prophet_mae, tech_prophet_rmse]
tech_statistics = tech_statistics.style.set_table_styles([cell_hover, index_names, headers])


# In[14]:


apple_statistics = pd.DataFrame(index=['MSE', 'MAE','RMSE'])
apple_statistics['Linear'] = [apple_linear_mse, apple_linear_mae, apple_linear_rmse]
apple_statistics['ARIMA'] = [apple_arima_mse, apple_arima_mae, apple_arima_rmse]
apple_statistics['Prophet'] = [apple_prophet_mse, apple_prophet_mae, apple_prophet_rmse]
apple_statistics = apple_statistics.style.set_table_styles([cell_hover, index_names, headers])


# HTML Builder

# In[15]:


style = '<link rel="stylesheet" href="style.css">'
header = '<h1> Final Project Overview </h1>'
tech_sector_header = '<h1> Tech Sector </h1>'
tech_sector_closing_blurb =''
tech_sector_data_blurb = ''
tech_sector_trend_blurb = ''
tech_sector_dist_blurb = ''
tech_sector_linear_blurb = ''
tech_sector_arima_blurb = ''
tech_sector_prophet_blurb = ''
tech_error_header = '<h3> Error Statistics </h3>'
apple_sector_header = '<h1> Apple Stock </h1>'
apple_sector_closing_blurb =''
apple_sector_data_blurb = ''
apple_sector_trend_blurb = ''
apple_sector_dist_blurb = ''
apple_sector_linear_blurb = ''
apple_sector_arima_blurb = ''
apple_sector_prophet_blurb = ''
apple_error_header = '<h3> Error Statistics </h3>'
sub_header = '<h2> An Exploratory Analysis of Stock Prediction </h2>'
topic_header = '<h3> Why this topic? </h3>'
topic_paragraph = '<p> We wanted to predict something relevant to the our economy. After analyzing various data sets, we decided we wanted to better understand the S&P 500. We each picked an individual sector and a corresponding stock in the S&P 500 to compute analysis and predictions over </p>'
data_header = '<h3> Data Exploration </h3>'
data_paragraph = '<p> The data contains daily closing prices of the S&P 500 from 2019 to 2020. The data was then broken down into an individual sector and a corresponding stock. These were: </p><ul><li> Tech Sector: Apple </li><li> Consumer Staples Sector: Kellogg </li><li> Consumer Discretionary Sector: Nike </li><li> Energy Sector: Occidential Petroleum </li><li> Industrials Sector: CH Robinson <li></ul><p> Then to get a better understanding of the data three charts were made: a linear graph of the stock/sectors daily closing price history, a trend/seasonlity plot, and a distribution plot. These can be seen with descriptions on the corresponding stock/sector pages. </p>'
data_analysis_header = '<h3> Data Analysis </h3>'
data_analysis_paragraph = '<p> Our Selected topic entails using Machine Learning to predict future stock prices based on historical data. The Machine Learn


# In[16]:


nav_bar2 = '<div class="topnav"><a href="../overview.html"> Overview </a><div class="dropdown"><button class="dropbtn">Tech</button><div class="dropdown-content"><a href="tech_report.html">Tech Sector</a><a href="apple_report.html">Apple</a></div></div><div class="dropdown"><button class="dropbtn">Consumer Staples</button><div class="dropdown-content"><a href="consumer_staples_report.html">Consumer Staples Sector</a><a href="kellogg_report.html">Kellogg</a></div></div><div class="dropdown"><button class="dropbtn">Consumer Discretionary</button><div class="dropdown-content"><a href="consumer_discretionary_report.html">Consumer Discretionary Sector</a><a href="nike_report.html">Nike</a></div></div><div class="dropdown"><button class="dropbtn">Industrial</button><div class="dropdown-content"><a href="industrial_report.html">Industrial Sector</a><a href="ch_robinson_report.html">CH Robinson</a></div></div><div class="dropdown"><button class="dropbtn">Energy</button><div class="dropdown-content"><a href="energy_report.html">Energy Sector</a><a href="occidential_report.html">Occidential</a></div></div></div>'
nav_bar = '<div class="topnav"><a href="overview.html"> Overview </a><div class="dropdown"><button class="dropbtn">Tech</button><div class="dropdown-con


# In[17]:


content_overview = style + nav_bar + '<br>' + header + sub_header + '<br>' + topic_header + topic_paragraph + '<br>' + data_header + data_paragraph +'<br>' + data_analysis_header + data_analysis_paragraph
html_overview = content_overview
with open('overview_report.html', 'w+') as file: file.write(html_overview)


# In[18]:


content_tech = style + nav_bar2 + '<br>' + tech_sector_header + '<br><div align="center">' + tech_closing_figure.to_html() + tech_sector_closing_blurb + '</div><br><div align="center>'+ tech_test_train_fig.to_html() + tech_sector_data_blurb + '</div><br><div align="center">'+ tech_linear_regression.to_html() + tech_sector_linear_blurb + '</div><br><div align="center">' + tech_arima.to_html() + tech_sector_arima_blurb + '</div><br><div align="center">' + tech_prophet.to_html() + tech_sector_prophet_blurb + '</div><br>'+ tech_error_header + tech_statistics.to_html()
html_tech = content_tech
with open('tech_report.html', 'w+') as file: file.write(html_tech)


# In[19]:


content_apple = style + nav_bar2 + '<br>' + apple_sector_header + '<br><div align="center">' + apple_closing_figure.to_html() + apple_sector_closing_blurb + '</div><br><div align="center>'+ apple_test_train_fig.to_html() + apple_sector_data_blurb + '</div><br><div align="center">'+ apple_linear_regression.to_html() + apple_sector_linear_blurb + '</div><br><div align="center">' + apple_arima.to_html() + apple_sector_arima_blurb + '</div><br><div align="center">' + apple_prophet.to_html() + apple_sector_prophet_blurb + '</div><br><div align="center>'+ apple_error_header + apple_statistics.to_html() + '</div>'
html_apple = content_apple
with open('apple_report.html', 'w+') as file: file.write(html_apple)


# In[ ]:




