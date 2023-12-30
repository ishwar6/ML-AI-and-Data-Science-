import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import base64
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from scipy.stats import skew, pearsonr

# Load the dataset

train = pd.read_csv("/usr/local/notebooks/datasets/Regression_Competition_Dataset/train.csv")
test = pd.read_csv("/usr/local/notebooks/datasets/Regression_Competition_Dataset/test.csv")

# check the data
train.head()
test.head()


'''Dimensions of train and test data'''
print('Dimensions of train data:', train.shape)
print('Dimensions of test data:', test.shape)

# Dimensions of train data: (1460, 81)
# Dimensions of test data: (1459, 80)

train.columns.values
# array(['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
#        'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
#        'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
#        'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond',
#        'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl',
#        'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
#        'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
#        'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
#        'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
#        'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
#        'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
#        'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
#        'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu',
#        'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars',
#        'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive',
#        'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
#        'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature',
#        'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition',
#        'SalePrice'], dtype=object)

test.columns.values

# array(['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
#        'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
#        'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
#        'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond',
#        'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl',
#        'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
#        'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
#        'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
#        'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
#        'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
#        'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
#        'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
#        'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu',
#        'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars',
#        'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive',
#        'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
#        'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature',
#        'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition'],
#       dtype=object)


"""Let's merge the train and test data and inspect the data type"""
merged = pd.concat([train, test], axis=0, sort=True)
print(merged.dtypes.value_counts())
print('Dimensions of data:', merged.shape)

# object     43
# int64      26
# float64    12
# dtype: int64
# Dimensions of data: (2919, 81)
# We can see from above

# There are 1460 instances in the train dataset
# There are 1459 instances in the test dataset
# There are 81 columns in the train dataset with one column "SalePrice", which is the target to be predicted.
# There are 80 columns in the test dataset. It doesn't contains the "SalePrice" column.
# 43 variables have the type object indicating that they are categorical variables which can be nominal or ordinal.
# 26 variables have the type int64 indicating they are numerical variables.
# 12 variables have the type float64 indicating they are numerical variables.


######################### Numeric Variable Analysis ###################################


del train["Id"] # we dont need it, it is unique id in each row

# get the numeric only from train dataset
df_num = train.select_dtypes(include = ['int64', 'float64'])

print(df_num.columns.values)
# ['MSSubClass' 'LotFrontage' 'LotArea' 'OverallQual' 'OverallCond'
#  'YearBuilt' 'YearRemodAdd' 'MasVnrArea' 'BsmtFinSF1' 'BsmtFinSF2'
#  'BsmtUnfSF' 'TotalBsmtSF' '1stFlrSF' '2ndFlrSF' 'LowQualFinSF'
#  'GrLivArea' 'BsmtFullBath' 'BsmtHalfBath' 'FullBath' 'HalfBath'
#  'BedroomAbvGr' 'KitchenAbvGr' 'TotRmsAbvGrd' 'Fireplaces' 'GarageYrBlt'
#  'GarageCars' 'GarageArea' 'WoodDeckSF' 'OpenPorchSF' 'EnclosedPorch'
#  '3SsnPorch' 'ScreenPorch' 'PoolArea' 'MiscVal' 'MoSold' 'YrSold'
#  'SalePrice']

print(len(df_num.columns.values))
# 37

# we have total 37 numeric variables. Since Id have been removed. 

# Univariate analysis involves the examination of a single variable in isolation. 
# It aims to understand the distribution, summary statistics, and characteristics of a single variable without considering its relationship with other variables.


first_phase_histograms = ["OverallQual", "OverallCond", "LowQualFinSF", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath",
 "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars", "OpenPorchSF", "EnclosedPorch",
 "3SsnPorch", "ScreenPorch", "MoSold"]

df_num[first_phase_histograms].hist(figsize=(20, 26), bins=40, color='green',alpha=0.5)


# Second Phase: to keep clutter minimum
second_phase_histograms = ["1stFlrSF", "2ndFlrSF", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "GarageArea", "GarageYrBlt", "GrLivArea", "LotArea", "LotFrontage", "MSSubClass", "YrSold", "YearBuilt", "YearRemodAdd", "WoodDeckSF", "TotalBsmtSF", "PoolArea", "SalePrice", "MasVnrArea", "MiscVal"]
df_num[second_phase_histograms].hist(figsize=(20, 26), bins=40, xrot=70, color = 'green',alpha=0.5)
















