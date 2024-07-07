#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# Sample data
data = {
    'Feature1': np.random.normal(loc=0, scale=1, size=1000),
    'Feature2': np.random.normal(loc=5, scale=2, size=1000),
    'Feature3': np.random.normal(loc=-3, scale=3, size=1000),
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Displaying the original DataFrame
print("Original DataFrame:")
df.head()


# In[2]:


# Simple random sampling
sample_df = df.sample(n=100)

# Displaying the sampled DataFrame
print("\nSampled DataFrame:")
sample_df.head()


# In[3]:


# Aggregating by taking the mean over intervals
aggregated_df = df.groupby(pd.cut(df.index, bins=10)).mean()

# Displaying the aggregated DataFrame
print("\nAggregated DataFrame:")
aggregated_df.head()


# In[4]:


from sklearn.decomposition import PCA

# Dimensionality reduction using PCA
pca = PCA(n_components=2)
pca_df = pca.fit_transform(df)

# Creating DataFrame from PCA results
pca_df = pd.DataFrame(data=pca_df, columns=['PC1', 'PC2'])

# Displaying the PCA-transformed DataFrame
print("\nPCA-transformed DataFrame:")
pca_df.head()


# In[ ]:




