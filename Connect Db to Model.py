#!/usr/bin/env python
# coding: utf-8

# Create a connection to database

# In[1]:


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
dataframe = psql.read_sql('SELECT * FROM tech_sector_export', conn)
print(dataframe)

# Close curser
cur.close()

# Close connection
conn.close()


# In[ ]:




