#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# Sample data
data = {
    'Value': np.random.randint(0, 100, size=100)
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Displaying the original DataFrame
print("Original DataFrame:")
df.head()


# In[2]:


# Discretization using equal-width binning
num_bins = 5
df['Value_Binned'] = pd.cut(df['Value'], bins=num_bins, labels=False)

# Displaying the discretized DataFrame
print("\nDiscretized DataFrame:")
df.head()


# In[ ]:




