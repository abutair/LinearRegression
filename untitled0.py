# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 12:40:30 2020

@author: mohammed abutair
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

 # Importing the dataset
 
dataset = pd.read_csv('Salary_Data.csv')

x = dataset.iloc[:,:-1].values

y = dataset.iloc[:,1].values


from sklearn.model_selection  import train_test_split
x_train , x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state =0)

# fiting the simple linear regression to the test
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)  # this line found the best fited line 

# prediciting the test set result
y_pred = regressor.predict(x_test)
y_pred_train = regressor.predict(x_train)

#visualizing the traning set result 
plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,y_pred_train,color='blue')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('salary vs exprince(training set)')
plt.xlabel('years of exprince')
plt.ylabel('salary')
plt.show()

