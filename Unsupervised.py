#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
data = np.random.randn(100, 2) * 2 + np.array([5, 5])

# Creating a DataFrame
df = pd.DataFrame(data, columns=['X', 'Y'])

# Displaying the original data
plt.scatter(df['X'], df['Y'])
plt.title('Original Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# In[2]:


# Using K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df)

# Adding cluster labels to the DataFrame
df['Cluster'] = kmeans.labels_

# Displaying the clustered data
plt.scatter(df['X'], df['Y'], c=df['Cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='red', s=200, label='Centroids')
plt.title('Clustered Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()


# In[ ]:




