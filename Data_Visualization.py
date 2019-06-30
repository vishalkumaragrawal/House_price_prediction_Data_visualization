# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 22:13:47 2019

@author: 1506284
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

#Finding the number of null values
train_data.info()
train_data.isnull().sum()

#finding the percentage of the missing vaalues column wise
miss = train_data.isnull().sum()/len(train_data)
miss= miss[miss>0]
miss.sort_values(inplace = True)
miss


#now we will se the target variable that is the salesprice
sns.distplot(train_data['SalePrice'])
print("skewness of the target varible is : ",train_data['SalePrice'].skew())
"""
So we found that the target variable is right skewd so we will apply log transform to it so that our prediction is more beter
By applying log transform we will make it normally distributed

"""
#applying log transform
target = np.log(train_data['SalePrice'])
print("After log transform the skewness in the target varible become :",target.skew())
sns.distplot(target)

#Now we will separeate the numeric and categorical data so that the visuallization is more easy

numerical_data = train_data.select_dtypes(include=[np.number])
categorical_data = train_data.select_dtypes(exclude=[np.number])
print("There are {} numerical columns and {} categoricaal columns".format(numerical_data.shape[1],categorical_data.shape[1]))

#we should remove the ID columns
numerical_data.drop("Id",axis = 1,inplace = True)


#now finding the corelation between the numerical columns

sns.heatmap(numerical_data.corr(),cmap='coolwarm')
plt.savefig('Heatmap')

corr=numerical_data.corr()
print(corr['SalePrice'].sort_values(ascending = False)[0:15])  #first 15
print("\n--------------------------------------------------\n")
print(corr['SalePrice'].sort_values(ascending = False)[-5:]) #last 5

"""
So we found that OverrAllQual is more corelated to saleprice so analysing it
"""

train_data['OverallQual'].unique()
#so it have value range from 1 to 10

# finding the salesprice median aorrosponding to each quality value
pivot_tab = train_data.pivot_table(index = 'OverallQual',values = 'SalePrice',aggfunc = 'median')
pivot_tab.plot(kind = 'bar',color = 'red')

"""
As the quality increases the sale price increases
which make sense so the variable is most co related and we are going in right direction
"""

#Now we are going to analyse the second most related field that is GrLivArea
sns.jointplot(train_data['GrLivArea'],train_data['SalePrice'],data = train_data)

#Now analyse the 3rd one GarageCars
train_data['GarageCars'].unique()
#So it is categical data 
sns.jointplot(train_data['GarageCars'],train_data['SalePrice'],data = train_data)  #this is of no use for categorical data

sns.barplot(train_data['GarageCars'],train_data['SalePrice'],data = train_data)
"""
So we find that people prefer more with gradge of size of 3 car capacity
"""

#now analysisng the 4th column GarageArea
train_data['GarageArea'].unique()
sns.jointplot(train_data['GarageArea'],train_data['SalePrice'],data = train_data)
#so it also have a linear relationship with the target variable SalePrice



"""

NOW WE WILL ANALYSE SOME OF THE CATEGORICAL COLUMNS
"""
categorical_data.describe()
categorical_data.info()

"""
Now we will find the median salesprice currosponding to the salescondition
"""
sales_condition_pivot_table = train_data.pivot_table(index = 'SaleCondition',values = 'SalePrice',aggfunc = 'median')
sales_condition_pivot_table.plot(kind='bar',color='red')

#so there is no normal patern in this 

#now as we found the corelation table for numericaal _data similarrly we use alloha test to find the significance of the the coloumns
#Doing the annova test
from scipy import stats
cat = [f for f in train_data.columns if train_data.dtypes[f] == 'object']
def anova(frame):
    anv = pd.DataFrame()
    anv['features'] = cat
    pvals = []
    for c in cat:
           samples = []
           for cls in frame[c].unique():
                  s = frame[frame[c] == cls]['SalePrice'].values
                  samples.append(s)
           pval = stats.f_oneway(*samples)[1]
           pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')

categorical_data['SalePrice'] = train_data.SalePrice.values
k = anova(categorical_data) 
k['disparity'] = np.log(1./k['pval'].values) 
sns.barplot(data=k, x = 'features', y='disparity') 
plt.xticks(rotation=90) 
plt.tightlayout()
plt 

"""
Now we will plt for the numerical and categorical coloumn
for numerical column we will plot distrivution plot to find id there is any column which is right skewd
And
for categorical column we will plot box plot to find if there exists any outlier
"""

num = [f for f in train_data.columns if train_data.dtypes[f] != 'object']
num.remove('Id')
nd = pd.melt(train_data, value_vars = num)
n1 = sns.FacetGrid (nd, col='variable', col_wrap=4, sharex=False, sharey = False)
n1 = n1.map(sns.distplot, 'value')
n1

#we found that most of the columns are right skewd

#now for the categorical data using box plot

def boxplot(x,y,**kwargs):
            sns.boxplot(x=x,y=y)
            x = plt.xticks(rotation=90)

cat = [f for f in train_data.columns if train_data.dtypes[f] == 'object']

p = pd.melt(train_data, id_vars='SalePrice', value_vars=cat)
g = sns.FacetGrid (p, col='variable', col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(boxplot, 'value','SalePrice')
g


#from this we found that it contains many outlier but we will not remove it bz it require many time 
#so we will leave it on the algorithms
"""


At last we found the the factors that are to be taken concern at data pre processing stage
1. removing the missing data
2. removing skewness
3. removing outliers

"""