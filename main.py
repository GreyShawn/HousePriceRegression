#invite people for the Kaggle party
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.stats import PearsonRConstantInputWarning
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('train.csv')
var = "GrLivArea"
var2 = "TotalBsmtSF"
dataLiv = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
dataBsm = pd.concat([df_train['SalePrice'], df_train[var2]], axis=1)

print(df_train.columns)

print(df_train['SalePrice'].describe())
sns.displot(df_train['SalePrice'])
print("Skewness: %f" % df_train['SalePrice'].skew()) 
print("Kurtosis: %f" % df_train['SalePrice'].kurt())
dataLiv.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
dataBsm.plot.scatter(x=var2, y='SalePrice', ylim=(0,800000))
plt.show()