#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = iris.target

# Using logistic regression as the model with increased max_iter
model = LogisticRegression(max_iter=1000)

# Using RFE for feature selection
selector = RFE(model, n_features_to_select=2)
selector = selector.fit(X, y)

# Displaying selected features
selected_features = X.columns[selector.support_]
print("Selected Features:")
print(selected_features)


# In[ ]:




