# Implementing Simple Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/Advertising.csv")


df['total_spend'] = df['TV'] + df['radio'] + df['newspaper']
print(df.head())

# sns.scatterplot(data=df, x='total_spend', y ='sales')

sns.regplot(data=df, x='total_spend', y='sales')

X = df['total_spend']
y = df['sales']

np.polyfit(X, y, 1)
spends = np.linspace(0,500,100)
predicted = 0.04868*spends + 4.24
plt.plot(spends, predicted, color='red')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


# from sklearn.model_family import ModelAlgo
# mymodel = ModelAlgo(p1, p2)
# mymodel.fit(X_train, y_train)
# predications = mymodel.predict(X_test)

# from sklearn.metrics import error_metric
# performance = error_metric(y_test, predictions)








