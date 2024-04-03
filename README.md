# House Price Prediction using Machine Learning in Python
We all have experienced a time when we have to look up for a new house to buy. 
But then the journey begins with a lot of frauds, negotiating deals, researching the local areas and so on.

# Importing Libraries and Dataset
Here we are using 

Pandas – To load the Dataframe
Matplotlib – To visualize the data features i.e. barplot
Seaborn – To see the correlation between features using heatmap

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

dataset = pd.read_excel("HousePricePrediction.xlsx")

print(dataset.head(5))

![Screenshot 2024-04-04 003625](https://github.com/bharathakshitha/mlproject/assets/137396041/836558d2-57b1-4d0e-9ee3-c27c8eb54dea)

dataset.shape

![Screenshot 2024-04-04 004023](https://github.com/bharathakshitha/mlproject/assets/137396041/836fd50c-59a1-4ff9-8f65-a23d7b621db6)

# Data Preprocessing
Now, we categorize the features depending on their datatype (int, float, object) and then calculate the number of them. 

obj = (dataset.dtypes == 'object')

object_cols = list(obj[obj].index)

print("Categorical variables:",len(object_cols))

 
int_ = (dataset.dtypes == 'int')

num_cols = list(int_[int_].index)

print("Integer variables:",len(num_cols))

fl = (dataset.dtypes == 'float')

fl_cols = list(fl[fl].index)

print("Float variables:",len(fl_cols))

 ![Screenshot 2024-04-04 004406](https://github.com/bharathakshitha/mlproject/assets/137396041/b23b03a4-1b0c-42d1-817b-ca03f5cbd494)
 

# Exploratory Data Analysis
EDA refers to the deep analysis of data so as to discover different patterns and spot anomalies. 
Before making inferences from data it is essential to examine all your variables.

So here let’s make a heatmap using seaborn library.

plt.figure(figsize=(12, 6))

sns.heatmap(dataset.corr(),
            cmap = 'BrBG',
            fmt = '.2f',
            linewidths = 2,
            annot = True)
            
  ![Screenshot 2024-04-04 004620](https://github.com/bharathakshitha/mlproject/assets/137396041/454902bc-c738-4b83-95d4-828267577add)

          

To analyze the different categorical features. Let’s draw the barplot.

unique_values = []

for col in object_cols:

  unique_values.append(dataset[col].unique().size)
  
plt.figure(figsize=(10,6))

plt.title('No. Unique values of Categorical Features')

plt.xticks(rotation=90)

sns.barplot(x=object_cols,y=unique_values)


 ![Screenshot 2024-04-04 004656](https://github.com/bharathakshitha/mlproject/assets/137396041/30b4260c-bd27-4c12-beba-f429187436c1)



The plot shows that Exterior1st has around 16 unique categories and other features have around  6 unique categories. 
To findout the actual count of each category we can plot the bargraph of each four features separately.

plt.figure(figsize=(18, 36))

plt.title('Categorical Features: Distribution')

plt.xticks(rotation=90)

index = 1
 
for col in object_cols:
    y = dataset[col].value_counts()
    plt.subplot(11, 4, index
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index), y=y)
    index += 1

 ![Screenshot 2024-04-04 004729](https://github.com/bharathakshitha/mlproject/assets/137396041/1a64a8cb-47f3-4fd9-a3f3-cb4b45a3531d)


# Data Cleaning
Data Cleaning is the way to improvise the data or remove incorrect, corrupted or irrelevant data.

As in our dataset, there are some columns that are not important and irrelevant for the model training. 
So, we can drop that column before training. There are 2 approaches to dealing with empty/null values

We can easily delete the column/row (if the feature or record is not much important).
Filling the empty slots with mean/mode/0/NA/etc. (depending on the dataset requirement).
As Id Column will not be participating in any prediction. So we can Drop it.

dataset.drop(['Id'],
             axis=1,
             inplace=True)
             

dataset['SalePrice'] = dataset['SalePrice'].fillna(

  dataset['SalePrice'].mean())
  

new_dataset = dataset.dropna()

new_dataset.isnull().sum()

![Screenshot 2024-04-04 004755](https://github.com/bharathakshitha/mlproject/assets/137396041/2f7eb08d-99e5-4cc4-bce1-27ea5f48f8c3)

OneHotEncoder – For Label categorical features
One hot Encoding is the best way to convert categorical data into binary vectors. This maps the values to integer values. By using OneHotEncoder, we can easily convert object data into int. So for that, firstly we have to collect all the features which have the object datatype. To do so, we will make a loop.

from sklearn.preprocessing import OneHotEncoder
 
s = (new_dataset.dtypes == 'object')

object_cols = list(s[s].index)

print("Categorical variables:")

print(object_cols)

print('No. of. categorical features: ', 
      len(object_cols))
      
![Screenshot 2024-04-04 004818](https://github.com/bharathakshitha/mlproject/assets/137396041/7a2de822-20e3-40ec-abdf-e51aa4b52546)


Then once we have a list of all the features. We can apply OneHotEncoding to the whole list.

OH_encoder = OneHotEncoder(sparse=False)

OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))

OH_cols.index = new_dataset.index

OH_cols.columns = OH_encoder.get_feature_names()

df_final = new_dataset.drop(object_cols, axis=1)

df_final = pd.concat([df_final, OH_cols], axis=1)




# Splitting Dataset into Training and Testing
X and Y splitting (i.e. Y is the SalePrice column and the rest of the other columns are X)

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split
 
X = df_final.drop(['SalePrice'], axis=1)

Y = df_final['SalePrice']
 
#Split the training set into

#training and validation set

X_train, X_valid, Y_train, Y_valid = train_test_split(
    X, Y, train_size=0.8, test_size=0.2, random_state=0)

# Model and Accuracy

from sklearn import svm

from sklearn.svm import SVC

from sklearn.metrics import mean_absolute_percentage_error
 
model_SVR = svm.SVR()

model_SVR.fit(X_train,Y_train)

Y_pred = model_SVR.predict(X_valid)
 
print(mean_absolute_percentage_error(Y_valid, Y_pred))

![Screenshot 2024-04-04 004843](https://github.com/bharathakshitha/mlproject/assets/137396041/cbc39283-5384-41e9-b622-4a8fe42aec17)


Random Forest Regression
Random Forest is an ensemble technique that uses multiple of decision trees and can be used for both regression and classification tasks.
To read more about random forests refer this.

from sklearn.ensemble import RandomForestRegressor
 
model_RFR = RandomForestRegressor(n_estimators=10)

model_RFR.fit(X_train, Y_train)

Y_pred = model_RFR.predict(X_valid)
 
mean_absolute_percentage_error(Y_valid, Y_pred)

![Screenshot 2024-04-04 004906](https://github.com/bharathakshitha/mlproject/assets/137396041/5ebca675-5a96-4988-9c42-14d43b2dda9d)


import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

np.random.seed(0)

num_samples = 100

num_features = 2

X = np.random.rand(num_samples, num_features)  # Features

true_coefficients = np.array([3, 5])  # True coefficients

noise = np.random.randn(num_samples)  # Noise

y = np.dot(X, true_coefficients) + noise  # Target variable (house prices)

#Define the mean squared error function

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

#Split the data into training and testing sets
split = int(0.8 * num_samples)

X_train, X_test = X[:split], X[split:]

y_train, y_test = y[:split], y[split:]

#Train a simple linear regression model
coefficients = np.linalg.lstsq(X_train, y_train, rcond=None)[0]


#Make predictions on the testing set

predictions = np.dot(X_test, coefficients)

#Calculate mean squared error

mse = mean_squared_error(y_test, predictions)

print("Mean Squared Error:",mse)

![Screenshot 2024-04-04 005021](https://github.com/bharathakshitha/mlproject/assets/137396041/e11b285b-f1d7-4be3-a8e6-198c0141506e)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

#Assuming you have your dataset loaded into features (X) and target variable (y)

#Split the data into training and testing sets
split = int(0.8 * num_samples)

X_train, X_test = X[:split], X[split:]

y_train, y_test = y[:split], y[split:]

#Train a linear regression model
model = LinearRegression()

model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)

#Calculate R-squared

r_squared = r2_score(y_test, y_pred)

print("R-squared:",r_squared)

![Screenshot 2024-04-04 005042](https://github.com/bharathakshitha/mlproject/assets/137396041/016ae739-6777-46e1-b3e6-3d0c4e493170)

