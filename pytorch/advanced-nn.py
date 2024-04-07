from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 

iris = load_iris()
X = iris['data']
y = iris['target']

 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1./3, random_state=1)
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# Lets do normalisation over X_train
X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
X_train_norm = torch.from_numpy(X_train_norm).float()
y_train = torch.from_numpy(y_train) 

train_ds = TensorDataset(X_train_norm, y_train)

torch.manual_seed(1)
batch_size = 2
train_dl = DataLoader(train_ds, batch_size, shuffle=True)


# Normalization
# X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
# This line normalizes the training data (X_train). Normalization is a preprocessing step to standardize the input data by subtracting the mean and dividing by the standard deviation. Here's why normalization is important:

# Improves Convergence: It helps the learning algorithm converge faster by ensuring that the feature values are on a similar scale. 
# Without normalization, features with larger numerical ranges could dominate the learning process, leading to slower convergence or suboptimal solutions.
# Reduces Skewness: It makes the training process less sensitive to the scale of features, allowing the model to learn more about the relationships between features and the target variable.
# Numerical Stability: It helps avoid numerical instabilities that can occur due to very large or very small values during computations.

# Broadcasting in NumPy
# The reason this operation still works correctly and applies the mean and standard deviation calculations to each element of X_train is due to a NumPy feature called broadcasting. 
# Broadcasting allows NumPy to work with arrays of different shapes when performing arithmetic operations. The smaller array is "broadcast" across the larger array so that they have compatible shapes.
