import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt 
import numpy as np
from scipy import stats

df=pd.read_csv('./data/auto3.csv')
print(df)
#print (df.columns)
#print (df.describe())
#print (df['drive-wheels'].value_counts())

"""dwc=df['drive-wheels'].value_c
ounts()# dwc=drive_wheels_count
print (dwc)
dwc.rename(colunms={'drive-wheels':'value_counts'}, inplace = True)
dwc.index.name='drive-wheels'
sns.boxplot (x="drive-wheels",y='price', data=df)
sns.boxplot(x='engine-size', y='price', data=df)""

print (df['engine-size'])
y=df['price']
x=df['engine-size']
plt.scatter(x,y)
plt.title('Scatterplot of Engine Size vs Price')
plt.xlabel("Engine Size")
plt.ylabel("Price")"""

#grouping
df_test=df[['drive-wheels','body-style','price']]
df_grp2=df_test.groupby(['drive-wheels'],as_index=False).mean()
df_grp=df_test.groupby(['drive-wheels','body-style'],as_index=False).mean()

#pivot
df_pivot=df_grp.pivot(index='drive-wheels',columns='body-style')
print (df_pivot)
df_pivot.replace(np.nan,20239.229524, inplace=True)

#heatmap
plt.pcolor(df_pivot, cmap='RdBu')
plt.colorbar()
plt.show()

#corralation 
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)#revisar

sns.regplot(x="highway-L/100km", y="price", data=df)
plt.ylim(0,)
"""
P-values:
    x<0.001 Strong certainty
    0.001<x<0.05 Moderate certainty
    0.5<x<0.1 weak certainty
    x>0.1 no certainty
"""

pearson_coef, p_values= stats.pearsonr(df['horsepower'],df['price'])
df=df.drop(['Unnamed: 0'], axis=1)
df.columns

#ANOVA
df_anova=df[['make', 'price']]
#group_anova=df_anova.groupby(['make'],as_index=False).mean()
group_anova=df_anova.groupby(['make'])
group_anova
analysis_anova_1=stats.f_oneway(group_anova.get_group('honda')['price'],group_anova.get_group('subaru')['price'])
analysis_anova_2=stats.f_oneway(group_anova.get_group('honda')['price'],group_anova.get_group('jaguar')['price'])
print (analysis_anova_1)
df.to_csv('./data/auto4.csv')
