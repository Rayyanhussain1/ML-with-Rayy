import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
Boston=pd.read_csv('Boston.csv')
print(Boston.head())
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
y=Boston[['medv']]
x=Boston[['crim']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
lr= LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_test.head())
print(y_pred[0:5])
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,y_pred)) 
x=Boston[['lstat']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
lr1=LinearRegression()
lr1.fit(x_train,y_train)
y_pred1=lr1.predict(x_test)
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,y_pred1))