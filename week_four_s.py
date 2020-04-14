# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 00:42:21 2020

@author: jrco1
"""

import pandas as pd
from sklearn.linear_model import LinearRegression 
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np

df=pd.read_csv('./data/auto4.csv')

lm=LinearRegression()#create a lienar regression object using the constructor 
y=df['price']#target
x=df[['highway-L/100km']] #variable predictor 
#x=df4
lm.fit(x,y)
yhat=lm.predict(x)
yhat
lm.intercept_
lm.coef_
price_estimate=lm.intercept_ +(lm.coef_*df['highway-L/100km'])

#df2['p2']=price_estimate.to_frame()

#multiple lineal model 
z=df[['horsepower','curb-weight','engine-size', 'highway-L/100km']]
lm.fit(z,df['price'])
Yhat=lm.predict(z)
Yhat
lm.intercept_
array=lm.coef_

sns.regplot(y=df['price'], x=df['highway-L/100km'], data=df)
plt.ylim(0,)


#residual plot - represents the error between the actual value 
sns.residplot(x=df['highway-L/100km'], y=df['price'], data=df)
plt.ylim(0,)

#distribution plots
ax1=sns.distplot(df['price'], hist=False, color ='r' , label='Actual Value')
ax2=sns.distplot(Yhat, hist=False, color='b', label='Fitted Values')
ax2=sns.distplot(Yhat, hist=False, color='b', label='Fitted Values', ax=ax1)


#calculate Polynomial of 3 order
x=df['highway-L/100km']
f=np.polyfit(x,y,3)
p=np.poly1d(f)#crete the model 
p
type (p)
f
print (np.poly1d(p), True)
print (np.poly1d(f), True)
p(50)
#f(50)
p.c
p.r
np.square(p)
#multi dimensional polynomial linear regression 
from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=2)
pr.fit_transform([[1,2]])
type(pr)
print (pr)

#Normalize the each feature simultaneously
from sklearn.preprocessing import StandardScaler
SCALE=StandardScaler()
SCALE.fit (df[['horsepower', 'highway-L/100km']])
x_scale=SCALE.transform(df[['horsepower', 'highway-L/100km']])


#Pipelines
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression 
from sklearn.pipeline import Pipeline

Input =[('scale', StandardScaler()),('polynomial',PolynomialFeatures(degree=2)),('model',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(df[['horsepower','curb-weight','engine-size','highway-L/100km']],y)
pipe
yhat2=pipe.predict(df[['horsepower','curb-weight','engine-size','highway-L/100km']])
ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
#ax3=sns.distplot(yhat2, hist=False, color='g', label='Second Fitted Values')
ax4=sns.distplot(yhat2, hist=False, color='y', label='Second fitted values', ax=ax1)
plt.show(ax4)

#evaluate how good the model fits on our data with:
#mean squared error 
#R-squared

from sklearn.metrics import mean_squared_error
mean_squared_error(df['price'],yhat) 
mean_squared_error(df['price'],Yhat)
mean_squared_error(df['price'],yhat2)

#r^2=(1-mse of regression line/mse of average of the data )
# if this values is close to 1 means the line id a good fit 
b=df[['highway-L/100km']]
c=df['price']
lm.fit(b,c)
R=lm.score(b,c)
R       

#let's predict of a car whit 30 highway
lm.predict(np.array(30.0).reshape(-1,1))
lm.coef_
lm.predict(np.array(-30.0).reshape(-1,1))

#to generate a sequence of values in a specified range
new_input=np.arange(1,101,2).reshape(-1,1)
lst_test=[[num for num in range(0,101,2)]]
yhat3=lm.predict(new_input)

df.to_csv('./data/auto5.csv', index=False)
