#1)Importing the libraries
import pandas as pd
import numpy as np

#2)reading the dataset
dataset= pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
#3)handeling null values cleaning the dataset
from sklearn.impute import SimpleImputer 
si=SimpleImputer(missing_values=0,strategy='mean')
X[:,:-1]=si.fit_transform(X[:,:-1])

#4)encoding the categorical variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder="passthrough")
X=ct.fit_transform(X)

#5)diving the data into training and test set 
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#6)Applying machine learning algorithm multiple linear regression 
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)


#PREDICT THE TEST DATA I.E. pROFIT
Y_pred=regressor.predict(X_test)

np.set_printoptions(precision=2)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1),(Y_test.reshape(len(Y_test),1))),axis=1))

#7)visualizing the data

import matplotlib.pyplot as  plt
plt.scatter(X_test,Y_test,color='blue')
plt.plot(Y_pred)
plt.xlabel('Investment')
plt.ylabel('pROFIT')
plt.show()