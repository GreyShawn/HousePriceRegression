#invite people for the Kaggle party
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.stats import PearsonRConstantInputWarning
import seaborn as sns
import numpy as np
from scipy.stats import norm
from seaborn.matrix import heatmap
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('train.csv')
print(df_train.columns)

#Facts about the most important variable

print(df_train['SalePrice'].describe())
#sns.displot(df_train['SalePrice'])
print("Skewness: %f" % df_train['SalePrice'].skew()) 
print("Kurtosis: %f" % df_train['SalePrice'].kurt())

#Relationship with numerical variables

var = "GrLivArea"
var2 = "TotalBsmtSF"
dataLiv = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
dataBsm = pd.concat([df_train['SalePrice'], df_train[var2]], axis=1)
#dataLiv.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
#dataBsm.plot.scatter(x=var2, y='SalePrice', ylim=(0,800000))

#Relationship withe categorical variables

varOverAllQual = 'OverallQual'
dataOverAllQual = pd.concat([df_train['SalePrice'], df_train[varOverAllQual]], axis=1)
#f,ax = plt.subplots(figsize=(8,6))
#fig = sns.boxplot(x=varOverAllQual, y="SalePrice", data = dataOverAllQual)
#fig.axis(ymin=0, ymax=800000)

varYearBuilt = 'YearBuilt'
dataYearBuilt = pd.concat([df_train['SalePrice'], df_train[varYearBuilt]],axis=1)
#f, ax =plt.subplots(figsize=(16,8))
#fig = sns.boxplot(x = varYearBuilt, y = "SalePrice", data = dataYearBuilt)
#plt.xticks(rotation = 90)

#Correlation matix(heatmap style)
corrmat = df_train.corr()
#f, ax = plt.subplots(figsize=(12,9))
#sns.heatmap(corrmat, vmax=.8, square=True)

#'SalePrice collerlation matrix (zoomed heatmap)'
k = 10 #number of variables
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
#cm = np.corrcoef(df_train[cols].values.T)
#sns.set(font_scale=1.25)
#hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size':10}, yticklabels = cols.values, xticklabels = cols.values)

#Scatter plots between 'SalePrice' and correlated variables 
sns.set()
colsScatter = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
#sns.pairplot(df_train[colsScatter], size = 2.0)

#missing data
total = df_train.isnull().sum().sort_values(ascending = False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat([total, percent], axis = 1, keys = ['Total','Percent'])
print(missing_data.head(20))

#dealing with missing data
#df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
#df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
#print(df_train.isnull().sum().max())

#Univariate anaysis / standardizing data
saleprice__scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])
low_range = saleprice__scaled[saleprice__scaled[:,0].argsort()][:10]
high_range = saleprice__scaled[saleprice__scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

#Testing Sale Price
#Search normality
#Histogram / normal probalitity plot(SalePrice)
df_train['SalePrice'] = np.log(df_train['SalePrice'])
#sns.distplot(df_train['SalePrice'], fit=norm)
#fig = plt.figure()
#res = stats.probplot(df_train['SalePrice'], plot = plt)

#Histogram / normal probalitity plot(GrLivarea)
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
#sns.distplot(df_train['GrLivArea'], fit = norm)
#fig = plt.figure()
#res = stats.probplot(df_train['GrLivArea'], plot = plt)

#Histogram / normal probalitity plot(TotalBsmtSF)
sns.distplot(df_train['TotalBsmtSF'], fit = norm)
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot = plt)
plt.show()