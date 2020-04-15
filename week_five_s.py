# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 15:36:46 2020

@author: jrco1
"""
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


#train test split function
df=pd.read_csv('./data/auto5.csv')
#df=pd.DataFrame(df)
x_train, x_test, y_train, y_test=train_test_split(df['highway-L/100km'], df['price'], test_size=0.3, random_state=0)

#each observation is used for both t4esting and training 
#let's use all data to testing and training 
lr=LinearRegression()
lr.fit(df[['highway-L/100km']],df['price'])
scores=cross_val_score(lr,df[['highway-L/100km']], df['price'],cv=3)
np.mean(scores)#R
scores
yhat5=cross_val_predict(lr,df[['highway-L/100km']], df['price'],cv=3)
ax1=sns.distplot(df['price'],hist=False, color='r', label='actual valor ')
sns.distplot(yhat5,hist=False,color ="b", label="new prediction", ax=ax1)

#how to pick the best order polynomial regression 
#we can calculate different R-squeared values as follows
rsqu_test=[]
order=list(range(1,10))
x_train=pd.DataFrame(x_train)
x_test=pd.DataFrame(x_test)
#dfind the scote for several grades
for grade in order:
    pr=PolynomialFeatures(degree=grade)
    x_train_pr=pr.fit_transform(x_train[['highway-L/100km']])
    x_test_pr=pr.fit_transform(x_test[['highway-L/100km']])
    lr.fit(x_train_pr, y_train)    
    rsqu_test.append(lr.score(x_test_pr,y_test))
    
#ridge regresion  (controls the magnitude of these polinomial coefficients )
RidgeModel=Ridge(alpha=0.1)
y=df['price']#target
x=df[['highway-L/100km']]
RidgeModel.fit(x,y)
yhat6=RidgeModel.predict(x)
ax1=sns.distplot(df['price'],hist=False, color='r', label='actual valor ')
ax2=sns.distplot(yhat6, hist=False,label=None, color='g', ax=ax1)
from sklearn.linear_model import LinearRegression 
lm=LinearRegression()
lm.fit(x,y)
R=lm.score(x,y)
R


#GRid search, explora  hiperparámetros (alpha) mediante validacion cruzada
parameters=[{'alpha':[0.001,0.1,1,10,100,1000,10000,100000,1000000],'normalize':[True,False]}]
rr=Ridge()
Grid1=GridSearchCV(rr,parameters, cv=4)
Grid1.fit(df[['horsepower', 'curb-weight', 'engine-size','highway-L/100km']], df['price'])
Grid1.best_estimator_
scores2=Grid1.cv_results_
scores2['mean_test_score']
scores2
scores2['param_alpha']
