#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pymc as pm
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


import pandas as pd
import os
import sklearn


# In[ ]:


# Import data
path_to_data = '../data/ml-32m/'
print(path_to_data)
ratings = pd.read_csv(os.path.join(path_to_data, 'ratings.csv'))
movies = pd.read_csv(os.path.join(path_to_data, 'movies.csv'))


# In[ ]:


print(ratings.head(3), '\n')
print(movies.head(3))


# In[ ]:


print(ratings.describe())


# In[ ]:


# Check for missing values
print(ratings.isnull().sum())
print(movies.isnull().sum())


# In[ ]:


# Normalise data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
ratings['scaled_rating'] = scaler.fit_transform(ratings[['rating']])


# In[ ]:


# Merge dataframes to make dataframe containing ratings and genres
full_ratings = pd.merge(ratings, movies, on='movieId')


# In[ ]:


print(full_ratings.head(3), '\n')


# In[ ]:


# Now group data by userID and genres, assign a mean rating for each (UserID, genre) pair then unpack these into a table
# Unfilled genres are filled with 0 rating

user_pref = full_ratings.groupby(['userId', 'genres']).rating.mean().unstack(fill_value=0)


# In[ ]:


# Now assume uniform prior for genres, use Dirichilet distribution as total prob sums to 1
with pm.Model() as model:
    user_genre_pref = pm.Dirichlet('user_genre_pref', a=np.ones(user_preferences.shape[1]))


# In[ ]:




