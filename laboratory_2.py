# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:10:46 2020

@author: jrco1
"""

import pandas as pd
import matplotlib.pylab as plt 
import numpy as np 


path='./data/auto2.csv'
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]


df=pd.read_csv(path,names=headers)
print (df)
print (df.head(n=5))
df.replace("?", np.nan, inplace = True)
#We replace "?" with NaN (Not a Number), which is Python's default missing value marker, for reasons of computational speed and convenience.
print (df)
missing_data= df.isnull() #method to identify missing values 
print (missing_data)#"True" stands for missing value, while "False" stands for not missing value

for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")
   
print(df['normalized-losses'].dtypes)
average=df['normalized-losses'].astype("float").mean(axis=0)#precaución porqupe al haber NAN no me permite realizar el premedio en dos lines separadas
print(df['normalized-losses'])
print(average)
df['normalized-losses'].replace(np.nan, average, inplace=True)
print(df['normalized-losses'])
print (df['bore'])
avg_bore=df['bore'].astype('float').mean(axis=0)
df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)

df['bore'].replace(np.nan,avg_bore, inplace=True)
print (df['bore'])
average_stroke=df['stroke'].astype('float').mean(axis=0)
df['stroke'].replace(np.nan, average_stroke, inplace=True)
print (df['stroke'].astype('float'))
average_horsepower=df['horsepower'].astype('float').mean(axis=0)
df['horsepower'].replace(np.nan, average_horsepower, inplace=True)

average_peak=df['peak-rpm'].astype('float').mean(axis=0)
df['peak-rpm'].replace(np.nan, average_peak, inplace =True)
print (df['peak-rpm'])
print (df['horsepower'].astype('int'))
print (df['num-of-doors'].value_counts())
#df['num-of-doors'].replace(np.nan, "four", inplace=True)
common=df['num-of-doors'].value_counts().idxmax()#we can also use the "idxmax()"method to calculate for us the most common types auromatically 
print (common)
df['num-of-doors'].replace(np.nan, common, inplace=True)
print (df['price'].head(20))
df.dropna(subset=["price"], axis=0, inplace=True)# simply drop whole row with NaN in "price" column
print (df['price'].head(20))
print (df.head())
df.reset_index(drop=True, inplace=True)
print (df.head())
print (df.dtypes)
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
df['city-L/100km'] = 235/df["city-mpg"]
df.rename(columns={'highway-mpg':'highway-L/100km'}, inplace=True)
print (df.dtypes)
print (df.columns)
df['length']=df['length']/df['length'].max()
df['width']=df['width']/df['width'].max()
df['height']=df['height']/df['height'].max()
print (df[['height','length']].head())
print (df['horsepower'].head())
df['horsepower']=df['horsepower'].astype(int, copy=True)
print (df['horsepower'].head())

import matplotlib as plt
from matplotlib import pyplot
"""plt.pyplot.hist(df['horsepower'])
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
print(bins)
group_names = ['Low', 'Medium', 'High']
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
print (df[['horsepower','horsepower-binned']].head(20))
print (df["horsepower-binned"].value_counts())
pyplot.bar(group_names, df["horsepower-binned"].value_counts())
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")"""
a = (0,1,2)

# draw historgram of attribute "horsepower" with bins = 3
plt.pyplot.hist(df["horsepower"], bins = 3)

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

dummy_variable_1=pd.get_dummies(df['fuel-type'])
print (dummy_variable_1.head())
print (df.columns)
dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'fuel-type-diesel':'fuel-type-diesel'}, inplace=True)
print (dummy_variable_1.head())
print (dummy_variable_1['fuel-type-gas'].value_counts())
print (dummy_variable_1.columns)
df['stroke'].isnull().value_counts()

df.to_csv('./data/auto3.csv')

df['width']
