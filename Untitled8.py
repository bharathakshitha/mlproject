#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
 


# In[2]:


dataset= pd.read_excel("HousePricePrediction.xlsx")


# In[3]:


dataset.head(5)


# In[4]:


dataset.shape


# In[5]:


obj = (dataset.dtypes == 'object')

object_cols = list(obj[obj].index)

print("Categorical variables:",len(object_cols))
 

int_ = (dataset.dtypes == 'int')

num_cols = list(int_[int_].index)

print("Integer variables:",len(num_cols))
 

fl = (dataset.dtypes == 'float')

fl_cols = list(fl[fl].index)

print("Float variables:",len(fl_cols))


# In[6]:


plt.figure(figsize=(12, 6))
sns.heatmap(dataset.corr(),

            cmap = 'BrBG',

            fmt = '.2f',

            linewidths = 2,

            annot = True)


# In[7]:


unique_values = []

for col in object_cols:

  unique_values.append(dataset[col].unique().size)

plt.figure(figsize=(10,6))

plt.title('No. Unique values of Categorical Features')

plt.xticks(rotation=90)

sns.barplot(x=object_cols,y=unique_values)


# In[8]:


plt.figure(figsize=(18, 36))

plt.title('Categorical Features: Distribution')

plt.xticks(rotation=90)

index = 1
 

for col in object_cols:

    y = dataset[col].value_counts()

    plt.subplot(11, 4, index)

    plt.xticks(rotation=90)

    sns.barplot(x=list(y.index), y=y)

    index += 1


# In[9]:


dataset.drop(['Id'],

             axis=1,

             inplace=True)


# In[10]:


dataset['SalePrice'] = dataset['SalePrice'].fillna(

  dataset['SalePrice'].mean())


# In[11]:


new_dataset = dataset.dropna()


# In[12]:



new_dataset.isnull().sum()


# In[13]:


from sklearn.preprocessing import OneHotEncoder


# In[14]:


s = (new_dataset.dtypes == 'object')

object_cols = list(s[s].index)

print("Categorical variables:")

print(object_cols)

print('No. of. categorical features: ', 

      len(object_cols))


# In[15]:


OH_encoder = OneHotEncoder(sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out()
df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)


# In[16]:


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']

# Split the training set into 
# training and validation set
X_train, X_valid, Y_train, Y_valid = train_test_split(
	X, Y, train_size=0.8, test_size=0.2, random_state=0)


# In[17]:


from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_percentage_error

model_SVR = svm.SVR()
model_SVR.fit(X_train,Y_train)
Y_pred = model_SVR.predict(X_valid)

print(mean_absolute_percentage_error(Y_valid, Y_pred))


# In[18]:


from sklearn.ensemble import RandomForestRegressor

model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train, Y_train)
Y_pred = model_RFR.predict(X_valid)

mean_absolute_percentage_error(Y_valid, Y_pred)


# In[23]:


# This code is contributed by @amartajisce
from catboost import CatBoostRegressor
cb_model = CatBoostRegressor()
cb_model.fit(X_train,y_train)
preds = cb_model.predict(X_valid) 

cb_r2_score=r2_score(Y_valid, preds)
cb_r2_score


# In[21]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate some example data
np.random.seed(0)
num_samples = 100
num_features = 2
X = np.random.rand(num_samples, num_features)  # Features
true_coefficients = np.array([3, 5])  # True coefficients
noise = np.random.randn(num_samples)  # Noise
y = np.dot(X, true_coefficients) + noise  # Target variable (house prices)

# Define the mean squared error function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Split the data into training and testing sets
split = int(0.8 * num_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train a simple linear regression model
coefficients = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

# Make predictions on the testing set
predictions = np.dot(X_test, coefficients)

# Calculate mean squared error
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:",mse)


# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Assuming you have your dataset loaded into features (X) and target variable (y)

# Split the data into training and testing sets
split = int(0.8 * num_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate R-squared
r_squared = r2_score(y_test, y_pred)
print("R-squared:",r_squared)


# In[ ]:




