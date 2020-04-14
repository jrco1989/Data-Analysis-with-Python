# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 23:40:19 2020

@author: jrco1
"""

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#Linear Regression and Multiple Linear Regression
#path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df=pd.read_csv('./data/auto4.csv')
df.head()
lm = LinearRegression()
x=df[['highway-L/100km']]
df.columns
y=df['price']
lm.fit(x,y)
Yhat=lm.predict(df[['highway-L/100km']])
Yhat[0:5]
lm.intercept_
lm.coef_
Z=df[['horsepower','curb-weight','engine-size','highway-L/100km']]
lm.fit(Z,df['price'])
lm.intercept_
lm.coef_
lm2=LinearRegression()
lm2.fit(df[['normalized-losses','highway-L/100km']], df['price'])
lm2.coef_

#visualization 
#Regression Plot
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-L/100km", y="price", data=df)
plt.ylim(0,)

plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)
df[["peak-rpm","highway-L/100km","price"]].corr()

#Residual plot
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df["highway-L/100km"], df["price"])
plt.show()

#Multiple linear regression 
Y_hat = lm.predict(Z)

plt.figure(figsize=(width, height))

ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)
sns.distplot(Y_hat, hist=False, color="g", label="Fitted Values2" , ax=ax1)

plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()

#Polynomial regression and Pipepline

#function to plot the data
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()

x = df['highway-L/100km']
y = df['price']
f = np.polyfit(x, y, 3) #Let's fit the polynomial using the function polyfit,
p = np.poly1d(f)
print(p)

PlotPolly(p, x, y, 'highway-mpg')
np.polyfit(x, y, 3)

f1 = np.polyfit(x, y, 11)
p1 = np.poly1d(f1)
PlotPolly(p1,x,y, 'Highway MPG')

#We create a PolynomialFeatures object of degree 2
pr=PolynomialFeatures(degree=2)
pr
Z_pr=pr.fit_transform(Z)
Z.shape
Z_pr.shape

#Pipeline
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe=Pipeline(Input)
pipe
pipe.fit(Z,y)
ypipe=pipe.predict(Z)
ypipe[0:4]

Input2=[('scale',StandardScaler()),('model',LinearRegression())]
pipe2=Pipeline(Input2)
pipe2.fit(Z,y)
ypipe2=pipe2.predict(Z)
ypipe2[0:10]
#sns.distplot(ypipe2, hist=False, color='b', label='other same fitted values',)

#Measures for In-Sample Evaluation
#highway_mpg_fit
X = df[['highway-L/100km']]
lm.fit(X, y)

# Find the R^2
print('The R-square is: ', lm.score(X, y))
Yhat5=lm.predict(X)
print('The output of the first four predicted value is: ', Yhat5[0:4])

#Mean Squared Error
mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)

#calculations for Multiple Linear Regression
# fit the model 
lm.fit(Z, df['price'])
# Find the R^2
print('The R-square is: ', lm.score(Z, df['price']))
Y_predict_multifit = lm.predict(Z)
print('The mean square error of price and predicted value using multifit is: ', mean_squared_error(df['price'], Y_predict_multifit))
#mean_squared_error(df['price'], Y_predict_multifit))

#Model 3: Polynomial Fit
#let’s import the function r2_score from the module metrics as we are using a different function    
r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)
mean_squared_error(df['price'], p(x))

#now we´ll use to method predict 
import matplotlib.pyplot as plt
import numpy as np
new_input=np.arange(1, 100, 1).reshape(-1, 1)
lm.fit(X, y)
lm
yhat6=lm.predict(new_input)
yhat6[0:5]
plt.plot(new_input, yhat6)
plt.show()

#Simple Linear Regression model (SLR) vs Multiple Linear Regression model (MLR)
#Simple Linear Model (SLR) vs Polynomial Fit
