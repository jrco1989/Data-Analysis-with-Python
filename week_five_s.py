# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 15:36:46 2020

@author: jrco1
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


#train test split function
df=pd.read_csv('./data/auto5.csv')
x_train, x_test, y_train, y_test=train_test_split(df['highway-L/100km'], df['price'], test_size=0.3, random_state=0)

#let's use all data to testing and training 
lr=LinearRegression()
lr.fit(df[['highway-L/100km']],df['price'])
scores=cross_val_score(lr,df[['highway-L/100km']], df['price'],cv=3)
np.mean(scores)#R

yhat5=cross_val_predict(lr,df[['highway-L/100km']], df['price'],cv=3)

#how to pick the best order polynomial regression 
#we can calculate different R-squeared values as follows
rsqu_test=[]
order=list(range(1,10))
"""for grade in order:
    pr=PolynomialFeatures(degree=grade)
    x_train_pr=pr.fit_transform(x_train[['highway-L/100km']])
    x_test_pr=pr.fit_transform(x_test[['highway-L/100km']])
    lr.fit(x_train_pr, y_train)    
    rsqu_test.append(lr.score(x_test_pr,y_test))
    """
#ridge regresion 
RidgeModel=Ridge(alpha=0.1)
RidgeModel.fit(x,y)
yhat6=RidgeModel.predict(x)


#GRid search
parameters=[{'alpha':[0.001,0.1,1,10,100,1000,10000,100000,1000000],'normalize':[True,False]}]
rr=Ridge()
Grid1=GridSearchCV(rr,parameters, cv=4)
Grid1.fit(df[['horsepower', 'curb-weight', 'engine-size','highway-L/100km']], df['price'])
Grid1.best_estimador_
scores2=Grid1.cv_results_
scores['mean_test_score']
