# Importing data manipulation and visualization libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importing preprocessing and machine learning tools
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
# Loading the dataset into a DataFrame
data = pd.read_csv("bank_sample.csv")

# Displaying the first few rows of the dataset
data.head()
# Displaying a summary of the DataFrame
data.info()
# Separating features and target variable
X = data.drop('y', axis=1)  # Dropping the target variable from features
y = data['y'].apply(lambda x: 1 if x == 'yes' else 0)  # Converting 'y' to numerical format
# Identifying categorical features
categorical_features = X.select_dtypes(include=['object']).columns

# Identifying numerical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
# Creating a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),  # Standardizing numerical features
        ('cat', OneHotEncoder(), categorical_features)  # One-hot encoding categorical features
    ])
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=1502)
# Applying preprocessing to training and test data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)
# Visualizing distributions of numerical features
for col in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()
# Checking for outliers in numerical features
for col in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=data[col])
    plt.title(f'Boxplot of {col}')
    plt.show()
# Analyzing distributions of categorical features
for col in categorical_features:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, data=data)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.show()
# Correlation analysis for numerical features
corr_matrix = data[numerical_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.show()
