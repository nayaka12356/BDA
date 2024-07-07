#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import Lasso

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = iris.target

# Using Lasso regularization for feature selection
lasso = Lasso(alpha=0.1)  # Set alpha parameter for regularization strength
lasso.fit(X, y)

# Displaying selected features (coefficients)
selected_features = X.columns[lasso.coef_ != 0]
print("Selected Features:")
print(selected_features)


# In[ ]:




