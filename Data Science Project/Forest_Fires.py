#!/usr/bin/env python
# coding: utf-8

# # Import Required Libraries and Dataset

# In[37]:


import numpy as np # NumPy is a Python library used for working with arrays.
                   #It also has functions for working in domain of linear algebra, fourier tranform, and matrices.
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle


# In[38]:


df = pd.read_csv("forestfires.csv")


# # Data Description

# In[39]:


df.head()


# In[40]:


# Aim:- We have to predict the burned areas in hectre.


# In[41]:


df.describe()


# In[42]:


df.info()


# In[43]:


# Checking for null values
df.isnull().any()


# In[44]:


sns.distplot(df["area"])


# In[45]:


df['area'] = np.log(df['area']+1) # The numpy.log() is a mathematical function that helps user to calculate Natural logarithm of x where x belongs to all the input array elements.
sns.distplot(df['area'])


# In[46]:


df.head()


# In[47]:


df['day'].unique()


# In[48]:


df['day'].value_counts()


# # Visualizing Dataset

# In[49]:


plt.rcParams['figure.figsize'] = [8,8]
day = sns.countplot(df['day'],order=['sun','mon','tue','wed','thu','fri','sat'])
day.set(title = 'Countplot for days in week', xlabel = 'days', ylabel = 'count')


# In[50]:


df.head()


# In[51]:


df['month'].unique()


# In[52]:


df['month'].value_counts()


# In[53]:


plt.rcParams['figure.figsize'] = [12,10]
day = sns.countplot(df['month'],order=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
day.set(title = 'Countplot for months in year', xlabel = 'months', ylabel = 'count')


# In[54]:


plt.figure(figsize=(15,15))
sns.heatmap(df.corr(),annot=True)


# In[55]:


# Now to further deal with data we need to convert the datatype of month and day from object-type to float.
# So we reform the dataset now,
df['month'].replace({'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12},inplace=True)
df['day'].replace({'sun':1,'mon':2,'tue':3,'wed':4,'thu':5,'fri':6,'sat':7},inplace=True)
df.head()


# In[56]:


df.info()


# # Test-Train Split

# In[57]:


y = df['area']         #  y is the target value.
x = df.drop(columns='area')  # x is the rest of the dataset except features.


# In[58]:


x


# In[59]:


y


# In[60]:


x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.25, random_state = 50)


# In[62]:


x_train


# In[63]:


x_test


# In[64]:


y_train


# In[65]:


y_test


# # MODELS

# ## Linear Regression

# In[67]:


lr = LinearRegression()
model = lr.fit(x_train,y_train)
lr_prediction = model.predict(x_test)
print(lr_prediction)


# In[69]:


print("Mean Squared Error = ",mse(lr_prediction,y_test))
print("Mean Absolute Error = ",mae(lr_prediction,y_test))
print("R2 Score = ",r2_score(lr_prediction,y_test))


# ## Random Forest Regression

# In[70]:


rfreg = RandomForestRegressor()
model = rfreg.fit(x_train,y_train)
rfreg_prediction = model.predict(x_test)
print(rfreg_prediction)


# In[71]:


print("Mean Squared Error = ",mse(rfreg_prediction,y_test))
print("Mean Absolute Error = ",mae(rfreg_prediction,y_test))
print("R2 Score = ",r2_score(rfreg_prediction,y_test))


# ## Decision Tree Regression

# In[72]:


dtr = DecisionTreeRegressor()
model = dtr.fit(x_train,y_train)
dtr_prediction = model.predict(x_test)
print(dtr_prediction)


# In[73]:


print("Mean Squared Error = ",mse(dtr_prediction,y_test))
print("Mean Absolute Error = ",mae(dtr_prediction,y_test))
print("R2 Score = ",r2_score(dtr_prediction,y_test))


# In[74]:


# As we can see that among the three models, least R2_Score is in Decision Tree Regression, so we use it for our model prediction.


# # Predicting Values

# In[76]:


answer = dtr.predict([[6,5,2,3,75.1,4.4,16.2,1.9,4.6,82,6.3,8]])
print(answer)


# In[77]:


pickle.dump(dtr,open('forest_fires.pkl','wb'))


# In[ ]:




