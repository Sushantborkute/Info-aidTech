#!/usr/bin/env python
# coding: utf-8

# Task 2:- Movie Recommendation System
# 
# Name :- Sushant Borkute

# loading libraries into the notebook

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#  Load the dataset into a Pandas dataframe

# In[2]:


df = pd.read_csv('movies.csv')
df.head()


# In[3]:


df.tail()


# In[4]:


df.drop(['genres'], axis=1)


# In[7]:


rate = pd.read_csv("ratings.csv")


# In[8]:


rate.columns


# In[9]:


rate=rate.loc[:,['userId', 'movieId', 'rating']]
rate.head()


# In[10]:


df=pd.merge(df,rate)


# In[11]:


df.shape


# In[12]:


df=df.iloc[:1000000]


# In[13]:


df.describe()


# In[14]:


df.groupby("title").mean()['rating'].sort_values(ascending=False)


# In[15]:


df.groupby("title").count()["rating"].sort_values(ascending=False)


# data frame in which we will have rating and number of ratings column

# In[16]:


ratings=pd.DataFrame(df.groupby("title").mean()['rating'])
ratings['number of ratings']=pd.DataFrame(df.groupby("title").count()["rating"])


# In[17]:


ratings.sort_values(by='rating',ascending=False)


# In[18]:


ratings.describe()


# In[19]:


plt.hist(ratings['rating'])
plt.show()


# In[20]:


plt.hist(ratings['number of ratings'],bins=50)
plt.show()


#  Making a pivot table

# In[21]:


pivot_table=df.pivot_table(index=["userId"],columns=["title"],values="rating")
pivot_table.head()


# In[22]:


pivot_table.shape


# In[23]:


def recommend_movie(movie):
    movie_watched=pivot_table[movie]
    similarity_movie=pivot_table.corrwith(movie_watched)
    #find the correlation between the movies
    similarity_movie=similarity_movie.sort_values(ascending=False)
    return similarity_movie.head()


# In[24]:


recommend_movie('Beautiful Girls (1996)')


# In[ ]:




