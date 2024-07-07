#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = iris.target

# Step 1: Filter-based feature selection (SelectKBest)
selector = SelectKBest(score_func=f_classif, k=3)
X_selected = selector.fit_transform(X, y)

# Step 2: Wrapper-based feature selection (RFECV)
model = LogisticRegression()
rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(5), scoring='accuracy')
X_final = rfecv.fit_transform(X_selected, y)

# Displaying selected features
selected_features = X.columns[selector.get_support()][rfecv.support_]
print("Selected Features:")
print(selected_features)


# In[ ]:




