#en linea d ecomando si corre, verificar"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats



#path  =('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv')
path=('./data/auto3.csv')
dfl = pd.read_csv(path)
dfl['stroke'].isnull().value_counts()
dfl['stroke'].replace(np.nan,dfl['stroke'].mean(axis=0), inplace=True)

dfl=dfl.drop(['Unnamed: 0'], axis=1)

corr_all=dfl.corr()#  we can calculate the correlation between variables of type "int64" or "float64" using the method "corr":
dfl[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr()
pearson_coef, p_values= stats.pearsonr(dfl[ 'stroke'],dfl['bore'])

sns.regplot(x='bore', y='price', data=dfl)
plt.ylim(0,)
dfl[['price','bore']].corr()

sns.regplot(x='engine-size', y='price', data=dfl)
print (plt.ylim(0,))
dfl[['engine-size', 'price']].corr()

#sns.regplot(x="highway-L/100km", y="price", data=dfl)
#dfl[["highway-L/100km", 'price']].corr()

sns.regplot(x="peak-rpm", y="price", data=dfl)
dfl[['peak-rpm','price']].corr()

sns.boxplot(x="drive-wheels", y="price", data=dfl)
sns.regplot(y='stroke',x='price', data=dfl)

sns.boxplot(x='body-style', y='price', data=dfl)
sns.boxplot(x="engine-location", y="price", data=dfl)

dfl.describe()
dfl.describe(include=['object'])

dfl['drive-wheels'].value_counts()
dfl['drive-wheels'].value_counts().to_frame()
drive_wheels_counts = dfl['drive-wheels'].value_counts().to_frame() #We can convert the series to a Dataframe as follows : 
drive_wheels_counts
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts
drive_wheels_counts.index.name = 'drive-wheels'
drive_wheels_counts

# engine-location as variable
engine_loc_counts = dfl['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)



#grouping 
dfl['drive-wheels'].unique()
df_group_one = dfl[['drive-wheels','body-style','price']]
df_group_one=df_group_one.groupby(['drive-wheels','body-style'], as_index=False).min()
df_group_one


df_gptest = dfl[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
grouped_test1

grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot

grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0
grouped_pivot

df_gptest2 = dfl[['body-style','price']]
grouped_test_bodystyle = df_gptest2.groupby(['body-style'],as_index= False).mean()
grouped_test_bodystyle

#heatmap 
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()

fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()

corr_total=dfl.corr()

pearson_coef, p_value = stats.pearsonr(dfl['wheel-base'], dfl['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  

pearson_coef, p_value = stats.pearsonr(dfl['horsepower'], dfl['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

pearson_coef, p_value = stats.pearsonr(dfl['length'], dfl['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

pearson_coef, p_value = stats.pearsonr(dfl['width'], dfl['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value ) 
dfl[['width','price','bore']].corr()
dfl['price']
dfl['width'].isnull().value_counts()
dfl['width']

pearson_coef, p_value = stats.pearsonr(dfl['curb-weight'], dfl['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  

grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
grouped_test2.head(2)

df_gptest

grouped_test2.get_group('4wd')['price']

# ANOVA
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val)   


f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val )


f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])  
   
print( "ANOVA results: F=", f_val, ", P =", p_val) 

f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])  

 
print("ANOVA results: F=", f_val, ", P =", p_val) 

dfl.to_csv('./data/auto4.csv', index=False)

df_c=dfl.corr()
df_c
plt.pcolor(df_c, cmap='RdBu')
plt.colorbar()
plt.show()

fig, ax = plt.subplots()
im = ax.pcolor(df_c, cmap='RdBu')

#label names
row_labels = df_c.columns.levels[1]
col_labels = df_c.index

#move ticks and labels to the center
ax.set_xticks(np.arange(df_c.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(df_c.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)
