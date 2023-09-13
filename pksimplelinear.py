# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 16:25:27 2020

@author: pv250022
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#df = pd.read_csv("C:\\Users\\pv250022\\Desktop\\MachineLearning-Python\\UdemyMachineLearnign\\TestData\\Salary_Data.csv")
df = pd.read_csv("D:\ExternalHArdriveDAta\TDofficedata\pkpythonpractice\customer1.csv")
X= df.iloc[:,0:1].values
y= df.iloc[:,-1].values
print(X)
print(y)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression

rgr =LinearRegression()
rgr.fit(X_train,y_train)

y_predict=rgr.predict(X_test)

plt.scatter(X_train,y_train,color ='red')
plt.plot(X_train,rgr.predict(X_train),color = 'blue')
plt.title("Train data graph")
plt.xlabel("Yrs of exp")
plt.ylabel("Salary")

plt.show()
         
plt.scatter(X_test,y_test,color ='orange')
plt.plot(X_test,y_predict,color = 'yellow')
plt.plot(X_train,rgr.predict(X_train),color = 'brown')
plt.title("Test data graph")
plt.xlabel("Yrs of exp")
plt.ylabel("Salary")
#plt.scatter(X_test,y_test,color ='blue')
#plt.plot(X_test,y_test,color ='red')
plt.show()