import numpy as np
import pandas as pd 


df=pd.read_csv('./data_analysis/data/auto.csv')
df=df.dropna(axis=1)
df['normalized-losses']
df['normalized-losses'].astype('float')
df.dropna() #to remove data that contains missing values, 
df.dropna(subset=['normalized-losses'], axis=0, inplace=True)
df.dropna(subset=['price'], axis=0, inplace=True)#inplace allows the modifiation to be done on the data set directly 
#df.replace(missing_value,new_value) method for replace missing values
mean=df['normalized-losses'].mean() #calculate the mean of the column 
df['normalized-losses'].replace([np.nan,'?'],mean)
df['city-mpg']=235/df['city-mpg']
df.rename(columns={'city-mpg':'city-L/100km'}, inplace=True) #renombrar una columna
df['city-L/100km']
df['price'].astype('object') #change th type of the column
df['price'].astype('float64')
df.drop(['city-L/100km'], axis=1)
df['city-L/100km']
         #you can choose to drop columns or rows that contain the missing values 
         #axis equal cero drops the enteri row
         #axis =1 drops the entire column 
df.dropna(subset=['price'], axis=0, inplace=True)#inplace allows the modifiation to be done on the data set directly 
#df.replace(missing_value,new_value) method for replace missing values
mean=df['normalized-losses'].mean() #calculate the mean of the column 
mean
df['normalized-losses'].replace(np.nan,mean)
df.rename(columns={'city-mpg':'city-L/100km'}, inplace=True) #renombrar una columna
df['city-L/100km']
df['price'].astype('object') #change th type of the column
df['price'].astype('float64')
df.drop(['city-L/100km'], axis=1)
#df.drop(['city-mpg'],axis=1)
df['length']
#methods of normalizing data 
#first is simple feature scaling:
#x_new=x_old/x_max
#the second method is called min-max 
#x_new=(x_old-x_min)/(x_max-x_min)
# the third methos is called z-score
#x_new=(x_old-u)/sigma where u is the average of the feature and sigma is the standard deviation
#df['length']=(df['length']-df['length'].mean())/df['lenght'].std()
print (df.columns)
bins=np.linspace(min(df['price']), max(df["price"]),4)#split the range in four intervals
group_names=["Low","Medium","High"]
df["price.binned"]=pd.cut(df["price"],bins,labels=group_names,include_lowest=True)#generate column division 

df.rename(columns={'city-mpg':'city-L/100km'}, inplace=True)
df.rename(columns={"price.binned":"price_binned"}, inplace=True)
print (df.columns)
print (df['fuel-system'])
print(df)
#pd.get_dummies(df['fuel']) #method for to convert categorical variables to dommy variables 
