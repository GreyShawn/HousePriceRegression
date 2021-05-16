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
print(df_train.columns)

df_train['SalePrice'].describe()
sns.displot(df_train['SalePrice'])
print("Skewness: %f" % df_train['SalePrice'].skew()) 
print("Kurtosis: %f" % df_train['SalePrice'].kurt())
print("Test commit")
plt.show()