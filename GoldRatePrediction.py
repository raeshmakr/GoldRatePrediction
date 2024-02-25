#!/usr/bin/env python
# coding: utf-8

# <b>Gold Rate Prediction</b>

# 
# <a href="https://colab.research.google.com/github/raeshmakr/GoldRatePrediction/blob/main/GoldRatePrediction.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Importing the Libraries for the project
# 

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# Data Collection and Process
# 

# In[11]:


# load the csv data to a Pandas DataFrame
gold_data = pd.read_csv('/content/gldRateData.csv')


# In[12]:


# print first 5 rows in the dataset
gold_data.head()


# In[13]:


# print last 5 rows of the dataset

gold_data.tail()


# In[14]:


# number of rows and columns
gold_data.shape


# In[15]:


# get some basic info about the data
gold_data.info()


# In[16]:


# checking the number of missing values
gold_data.isnull().sum()


# In[17]:


# getting the statistical measures of the data
gold_data.describe()


# Correlation:
# 1. Positive Correlation
# 2. Negative Correlation

# In[24]:


correlation = gold_data.corr(numeric_only=True)


# In[25]:


# constructing a heatmap to understand the correlatiom
plt.figure(figsize = (8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',annot=True, annot_kws={'size':8}, cmap='Blues')


# In[21]:


# correlation values of GOLD
print(correlation['GLD'])


# In[22]:


# checking the distribution of Gold Rate
sns.distplot(gold_data['GLD'],color='blue')


# Splitting the Features and Target

# In[26]:


X = gold_data.drop(['Date','GLD'],axis=1)
Y = gold_data['GLD']


# In[27]:


print(X)


# In[28]:


print(Y)


# Splitting into Training data and Test Data

# In[29]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)


# Model Training:
# Random Forest Regressor

# In[ ]:





# In[30]:


regressor = RandomForestRegressor(n_estimators=100)


# In[31]:


# training the model
regressor.fit(X_train,Y_train)


# Model Evaluation

# In[32]:


# prediction on Test Data
test_data_prediction = regressor.predict(X_test)


# In[37]:


print(test_data_prediction)


# In[34]:


# R squared error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ", error_score)


# Compare the Actual Values and Predicted Values in a Plot

# In[35]:


Y_test = list(Y_test)


# In[36]:


plt.plot(Y_test, color='blue', label = 'Actual Value')
plt.plot(test_data_prediction, color='red', label='Predicted Value')
plt.title('Actual Rate vs Predicted Rate')
plt.xlabel('Number of values')
plt.ylabel('GOLD Rate')
plt.legend()
plt.show()

