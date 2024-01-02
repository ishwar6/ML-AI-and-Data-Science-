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


################################## Numerical Feature Transformation #######################################################

# As we noticed from above Histograms that most of the Numerical Features do not obey the Gaussian Distribution 
# so transforming the Numerical Variables might give us the boost in the performace of the models. Transformation will happen in the following steps

# 1. We will be calculating the Skewness of Numberical Variables on the Training Dataset (train.csv). We will compute the Skewness of a column after dropping the missing values in it. 

# 2. We will filter out the Numerical Variables having Skewness greater than some threshhold say 0.75.

# 3. We will apply Log Transformation on all Numerical Variables in both the train and test files which are having the skewness greater than threshold.



all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index # Getting the Numerical Features
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) # Computing the Skewness of Columns

# skewed_feats

# MSSubClass        1.406210
# LotFrontage       2.160866
# LotArea          12.195142
# OverallQual       0.216721
# OverallCond       0.692355
# YearBuilt        -0.612831
# YearRemodAdd     -0.503044
# MasVnrArea        2.666326
# BsmtFinSF1        1.683771
# BsmtFinSF2        4.250888
# BsmtUnfSF         0.919323
# TotalBsmtSF       1.522688
# 1stFlrSF          1.375342
# 2ndFlrSF          0.812194
# LowQualFinSF      9.002080
# GrLivArea         1.365156
# BsmtFullBath      0.595454
# BsmtHalfBath      4.099186
# FullBath          0.036524
# HalfBath          0.675203
# BedroomAbvGr      0.211572
# KitchenAbvGr      4.483784
# TotRmsAbvGrd      0.675646
# Fireplaces        0.648898
# GarageYrBlt      -0.648708
# GarageCars       -0.342197
# GarageArea        0.179796
# WoodDeckSF        1.539792
# OpenPorchSF       2.361912
# EnclosedPorch     3.086696
# 3SsnPorch        10.293752
# ScreenPorch       4.117977
# PoolArea         14.813135
# MiscVal          24.451640
# MoSold            0.211835
# YrSold            0.096170
# dtype: float64


skewed_feats = skewed_feats[skewed_feats > 0.75] # Keeping only those having skewness greater than 0.75
skewed_feats = skewed_feats.index # Getting the columns in a separate list
print(skewed_feats)

# Index(['MSSubClass', 'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1',
#        'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
#        'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'KitchenAbvGr',
#        'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
#        'ScreenPorch', 'PoolArea', 'MiscVal'],
#       dtype='object')

all_data[skewed_feats] = np.log1p(all_data[skewed_feats]) #Applying the log transformation on the Chosen Numerical Features


############################### Categorical Variable Encoding ##########################################

all_data = pd.get_dummies(all_data) # It automatically transforms the Categorical Variables

# we can see how the SaleCondition which is having the possible values Normal, Abnorml, AdjLand, Alloca, Family and Partial,
# is now split into 6 different columns with a value of 1 present in the respective column indicating its presence in the respective instance.
# SaleCondition_Abnormal, SaleCondition_Normal, SaleCondition_Partial etc. 



###################################################  Percentage of Missing Values ###################################################  

def percent_missing(df):
    data = pd.DataFrame(df)
    df_cols = list(pd.DataFrame(data))
    dict_x = {}
    for i in range(0, len(df_cols)):
        dict_x.update({df_cols[i]: round(data[df_cols[i]].isnull().mean()*100,2)})
    
    return dict_x

missing = percent_missing(all_data)
df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)
print('Percent of missing data')
# df_miss[0:20]

# [('LotFrontage', 16.65),
#  ('GarageYrBlt', 5.45),
#  ('MasVnrArea', 0.79),
#  ('BsmtFullBath', 0.07),
#  ('BsmtHalfBath', 0.07),
#  ('BsmtFinSF1', 0.03),
#  ('BsmtFinSF2', 0.03),
#  ('BsmtUnfSF', 0.03),
#  ('TotalBsmtSF', 0.03),
#  ('GarageCars', 0.03),
#  ('GarageArea', 0.03),
#  ('MSSubClass', 0.0),
#  ('LotArea', 0.0),
#  ('OverallQual', 0.0),
#  ('OverallCond', 0.0),
#  ('YearBuilt', 0.0),
#  ('YearRemodAdd', 0.0),
#  ('1stFlrSF', 0.0),
#  ('2ndFlrSF', 0.0),
#  ('LowQualFinSF', 0.0)]


## Mean Imputation

# Now we will replace the missing values with the mean of the respective columns.

all_data = all_data.fillna(all_data.mean())


missing = percent_missing(all_data)
df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)
print('Percent of missing data')
# df_miss[0:20]
# Percent of missing data
# [('MSSubClass', 0.0),
#  ('LotFrontage', 0.0),
#  ('LotArea', 0.0),
#  ('OverallQual', 0.0),
#  ('OverallCond', 0.0),
#  ('YearBuilt', 0.0),
#  ('YearRemodAdd', 0.0),
#  ('MasVnrArea', 0.0),
#  ('BsmtFinSF1', 0.0),



#creating matrices for sklearn:
X_train = all_data[:train.shape[0]] # Retrieving the rows for train from the all_data
X_test = all_data[train.shape[0]:] # Retrieving the rows for test from the all_data
y_train = train.SalePrice # Retrieving the output variable "SalePrice" of the train dataset 



# Now we can use regression: 

# DummyRegressor
from sklearn.dummy import DummyRegressor
dummy_reg = DummyRegressor(strategy="mean")

#The strategy parameter is set to "mean". This means that the dummy regressor will make predictions by always using the mean (average) value of the target variable as the predicted value.
# it simply predicts the mean value for all examples.
dummy_reg.fit(X_train, y_train)

# Now predict

y_pred = dummy_reg.predict(X_train)

# lets compute the RMS: 
from sklearn.metrics import mean_squared_error
from math import sqrt

dummy_rmse = sqrt(mean_squared_error(y_train, y_pred))
print(dummy_rmse)

#79415.29

# Ridge Regression

from sklearn import linear_model
ridge_reg = linear_model.Ridge(alpha=0.5)
ridge_reg.fit(X_train, y_train)

y_pred = ridge_reg.predict(X_train)
# Computing the Root Mean Squared Error 
from sklearn.metrics import mean_squared_error
from math import sqrt
ridge_rmse = sqrt(mean_squared_error(y_train, y_pred))
print(ridge_rmse)



















