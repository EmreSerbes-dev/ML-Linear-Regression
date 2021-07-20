import numpy as np #For data
import matplotlib.pyplot as plt #For graphics
import pandas as pd #For data
from sklearn.model_selection import train_test_split #For splitting the dataset
from sklearn.linear_model import LinearRegression #For linear regression

dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:, :-1].values #All rows and all column without last column
y = dataset.iloc[:, -1].values  #All rows and only last column(dependent column)  # -1 is dependent column's index

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
# print(f"{x_train} \n")

regressor = LinearRegression()
regressor.fit(x_train, y_train) #1. parameter: independent variable vector, 2. parameter: dependent variable
# print(f"{x_train} \n")

y_pred = regressor.predict(x_test)

plt.scatter(x_train, y_train, color = "red") #Choosing x axis datas, y axis datas, and dot color
plt.plot(x_train, regressor.predict(x_train), color = "blue") #Choosing x axis datas, y axis datas, and line color
plt.title("Salary and Experience at Work") #Name of graphic
plt.xlabel("Experience at Work") #Title of x axis
plt.ylabel("Salary") #Title of y axis
plt.show() #For show the graphic