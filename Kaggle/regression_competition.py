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
second_phase_histograms = ["1stFlrSF", "2ndFlrSF", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "GarageArea", 
                           "GarageYrBlt", "GrLivArea", "LotArea", "LotFrontage", "MSSubClass", "YrSold", "YearBuilt", 
                           "YearRemodAdd", "WoodDeckSF", "TotalBsmtSF", "PoolArea", "SalePrice", "MasVnrArea", "MiscVal"]
df_num[second_phase_histograms].hist(figsize=(20, 26), bins=40, xrot=70, color = 'green',alpha=0.5)


# Findings
# Variables like "MoSold", "OverallQual", "TotRmsAbvGrd" looks more like Gaussian Variables. Variable description is given as below.
# TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
# Features such as "1stFlrSF", "TotalBsmtSF", "LotFrontage", "GrLiveArea" seems to share a similar distribution to the one we have with "SalePrice".
#This is a Key Indication that they can help in Modelling. Variable description is given as below.

# Applying Numerical Transformations like Log Transformations might help in improving the performance of the model as many variables don't obey the Gaussian Distribution.

############################ Bivariate #####################################

# Scatter Plot comes under the Bivariate analysis. It shows the relationship between the variables. 
# We will be plotting the scatter plot between the input variables and the target variable "SalePrice" which is to be predicted. 
 # We won't be plotting the scatter plot for the following input numerical variables 'MiscVal', 'MoSold', 'YrSold' 


numeric_columns = ["OverallQual", "OverallCond", "LowQualFinSF", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath",
 "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars", "OpenPorchSF", "EnclosedPorch",
 "3SsnPorch", "ScreenPorch", "1stFlrSF", "2ndFlrSF", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "GarageArea", "GarageYrBlt",
  "GrLivArea", "LotArea", "LotFrontage", "MSSubClass", "YearBuilt", "YearRemodAdd", "WoodDeckSF", "TotalBsmtSF", "PoolArea", "MasVnrArea"]


fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(18, 146))
for i, feature in enumerate(list(df_num[numeric_columns]), 1):
    plt.subplot(len(list(numeric_columns)), 2, i)
    sns.scatterplot(x=feature, y='SalePrice', data=df_num)
    plt.xlabel('{}'.format(feature), size=12,labelpad=12.5)
    plt.ylabel('SalePrice', size=12, labelpad=12.5)
plt.show()

## Findings

# 1. We can see that a lot of data points are located on x = 0 which may indicate the absence of such feature in the house. For example LowQualFinSF, OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch, 2ndFlrSF, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, GarageArea, WoodDeckSF, TotalBsmtSF, PoolArea and MasVnrArea. All are having some data points located on x=0, indicating missing values in the respective columns. Below are the description of the Variables for a given house.

# 2. The scatter plot between "TotalBsmtSF" and "SalePrice" seems to have a Linear relationship, which would be helpful in modelling. 

# 3. The scatter plot between "GrLivArea" and "SalePrice" seems to have a Linear relationship, which would be helpful in modelling. 

# * GrLivArea: GrLivArea: Above grade (ground) living area square feet

# 4. The scatter plot between "MasVnrArea" and "SalePrice" seems to have a Linear relationship, which would be helpful in modelling. 

# 5. 1ndFlrSF and 2ndFlrSF seems to have a Linear Relationship with the "SalePrice". 

# 6. There are some outliers in GrLivArea(Above grade (ground) living area square feet). Some houses with Large GrLivArea tend to have low prices. 


####################################### Correlation Measure with Target #######################################

# We have to use Pearson correlation: 


# The Pearson correlation measures the strength of the linear relationship between two variables. 
# It has a value between -1 to 1, with a value of -1 meaning a total negative linear correlation, 
# 0 being no correlation, and + 1 meaning a total positive correlation.

result = df_num.drop("SalePrice", axis=1).apply(lambda x: x.corr(df_num.SalePrice, "pearson"))
result = result.sort_values(kind='quicksort", ascending= False )
print(result)

# OverallQual      0.790982
# GrLivArea        0.708624
# GarageCars       0.640409
# GarageArea       0.623431
# TotalBsmtSF      0.613581
# 1stFlrSF         0.605852
# FullBath         0.560664
# TotRmsAbvGrd     0.533723
# YearBuilt        0.522897
# YearRemodAdd     0.507101
# GarageYrBlt      0.486362
# MasVnrArea       0.477493
# Fireplaces       0.466929
# BsmtFinSF1       0.386420
# LotFrontage      0.351799
# WoodDeckSF       0.324413
# 2ndFlrSF         0.319334
# OpenPorchSF      0.315856
# HalfBath         0.284108
# LotArea          0.263843
# BsmtFullBath     0.227122
# BsmtUnfSF        0.214479
# BedroomAbvGr     0.168213
# ScreenPorch      0.111447
# PoolArea         0.092404
# MoSold           0.046432
# 3SsnPorch        0.044584
# BsmtFinSF2      -0.011378
# BsmtHalfBath    -0.016844
# MiscVal         -0.021190
# LowQualFinSF    -0.025606
# YrSold          -0.028923
# OverallCond     -0.077856
# MSSubClass      -0.084284
# EnclosedPorch   -0.128578
# KitchenAbvGr    -0.135907
# dtype: float64

                            
# Findings
# We can see the top correlated variables with the "SalePrice" which have a strong correlation.

# These variables include, "OverallQual", "GrLivArea", "GarageCars", "GarageArea", "FullBath" and so on.

# OverallQual: Rates the overall material and finish of the house
# GrLivArea: Above grade (ground) living area square feet
# GarageCars: Size of garage in car capacity
# FullBath: Full bathrooms above grade
# These Features are the possible candidates of including as a feature for modelling.

# The number of Cars that fit into the garage is a consequence of the Garage Area. 'GarageCars' and 'GarageArea' are also some of the most strongly correlated variables. 
# This is a case of Multicollinearity. So one of the feature should be chosen, we will go with GarageCars as that is most correlated with the Saleprice.

########################## Categorical Variable Analysis #########################################
         
 # Function responsible for plotting the BoxPlot
def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)

# Replacing the Missing Values in the Categorical Variables with the "MISSING" string
def fillMissingCatColumns(data,categorical):
    for c in categorical:
        data[c] = data[c].astype('category')
        if data[c].isnull().any():
            data[c] = data[c].cat.add_categories(['MISSING'])
            data[c] = data[c].fillna('MISSING')
# Main function responsible for plotting the BoxPlots
def getboxPlots(data,var,categorical):
    fillMissingCatColumns(data,categorical)
    f = pd.melt(data, id_vars=var, value_vars=categorical)
    g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, height=5)
    g = g.map(boxplot, "value", var)


# this is the main client driver code: that will use above 3 functions. 

categorical = [f for f in train.columns if train.dtypes[f] == 'object']   
train_copy = train.copy(True) 
getboxPlots(train_copy,'SalePrice', categorical)

# Street, which is the type of road access to the the property, with Pave tend to have more price.
# ExterQual (exterior quality of the house) with excellent condition has more price. Same holds for BsmtQual, KichenQual, GarageQual and PoolQC.
# Houses with Central Air conditioning (CentralAir) has more price.
# Houses having PavedDrive are having more price.
# Houses with Foundation of Poured Contrete has more price.










                            

