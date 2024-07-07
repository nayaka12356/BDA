#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Sample data
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Gender': ['Female', 'Male', 'Male'],
    'Income': [50000, 60000, 70000]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Displaying the original DataFrame
print("Original DataFrame:")
df


# In[2]:


# Example of data transformation:
# 1. Filter data for people with age greater than 25
filtered_df = df[df['Age'] > 25]

# 2. Encode categorical variable 'Gender' using one-hot encoding
encoded_df = pd.get_dummies(df, columns=['Gender'])

# 3. Scale numerical variable 'Income' using Min-Max scaling
df['Scaled_Income'] = (df['Income'] - df['Income'].min()) / (df['Income'].max() - df['Income'].min())

# Displaying the transformed DataFrames
print("Filtered DataFrame (Age > 25):")
filtered_df


# In[3]:


print("Encoded DataFrame:")
encoded_df


# In[4]:


print("DataFrame with Scaled Income:")
df


# In[ ]:




