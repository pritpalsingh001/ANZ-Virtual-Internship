# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 00:22:06 2020

@author: HP
"""
#importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#reading dataset
dataset=pd.read_excel('ANZ synthesised transaction dataset.xlsx');

#getting dataset of salaries
cus_sal=dataset[dataset["txn_description"]=="PAY/SALARY"].groupby("customer_id").mean()

#annual salaries
salaries = []
for customer_id in dataset["customer_id"]:
    salaries.append(int(cus_sal.loc[customer_id]["amount"]))
    

dataset["annual_salary"] = salaries

#preparing data before applying regression
cust_data = dataset.groupby("customer_id").mean()
print("Mean annual salary by customer: ")
print(cust_data.head(), "\n")

#linear regression
X=cust_data.iloc[:,:-1]
y=cust_data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
linear_Reg=LinearRegression()
linear_Reg.fit(X_train,y_train)

#printing score of training data
print(f"Linear Regression Training Score: {linear_Reg.score(X_train, y_train)}\n")
print("Predictions using test data:")
#making predictions
print(linear_Reg.predict(X_test), "\n")

#printing score of testing data
print(f"Linear Regression Testing Score: {linear_Reg.score(X_test, y_test)}\n")


#making dataframe of actual and preicted values
y_test.shape
y_test=y_test.values.reshape(20,)

y_pred=linear_Reg.predict(X_test)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df

#plotting a bar graph
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()



#preparing data for decision tree model
dataset_dt=dataset[["txn_description", "gender", "age", "merchant_state", "movement"]]
pd.get_dummies(dataset_dt).head()

#making dummy attributes
X=pd.get_dummies(dataset_dt)
y=dataset["annual_salary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#decision tree classifier
from sklearn.tree import DecisionTreeClassifier ,DecisionTreeRegressor
classifier= DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)
print(y_pred)
classifier.score(X_train,y_train)
classifier.score(X_test,y_test)



#decision tree regressor
dt_reg=DecisionTreeRegressor()
dt_reg.fit(X_train,y_train)
dt_reg.score(X_train,y_train)


dt_reg.score(X_test,y_test)

y_test.shape
y_test.values.reshape(2409,)

y_pred=dt_reg.predict(X_test)
dt_d = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred.flatten()})
dt_d


df2 = dt_d.head(25)
df2.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()