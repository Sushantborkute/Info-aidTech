#!/usr/bin/env python
# coding: utf-8

# Task 3:- Credit Card Fraud Detection
# 
# Name :- Sushant Borkute

# IMPORTING LIBRARIES:

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import warnings
warnings.filterwarnings('ignore')


# READING DATASET :

# In[2]:


data=pd.read_csv('creditcard.csv')


# In[3]:


data.head()


# NULL VALUES:

# In[4]:


data.isnull().sum()


# INFORMATION

# In[5]:


data.info()


# DESCRIPTIVE STATISTICS

# In[6]:


data.describe().T.head()


# In[7]:


data.shape


# In[8]:


data.columns


# FRAUD CASES AND GENUINE CASES

# In[9]:


fraud_cases=len(data[data['Class']==1])


# In[10]:


print(' Number of Fraud Cases:',fraud_cases)


# In[11]:


non_fraud_cases=len(data[data['Class']==0])


# In[12]:


print('Number of Non Fraud Cases:',non_fraud_cases)


# In[13]:


fraud=data[data['Class']==1]


# In[14]:


genuine=data[data['Class']==0]


# In[15]:


fraud.Amount.describe()


# In[16]:


genuine.Amount.describe()


# EDA

# In[17]:


data.hist(figsize=(20,20),color='lime')
plt.show()


# In[18]:


rcParams['figure.figsize'] = 16, 8
f,(ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(genuine.Time, genuine.Amount)
ax2.set_title('Genuine')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()


# CORRELATION

# In[19]:


plt.figure(figsize=(10,8))
corr=data.corr()
sns.heatmap(corr,cmap='BuPu')


# Let us build our models:

# In[20]:


from sklearn.model_selection import train_test_split


# Model 1:

# In[21]:


X=data.drop(['Class'],axis=1)


# In[22]:


y=data['Class']


# In[23]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=123)


# In[24]:


from sklearn.ensemble import RandomForestClassifier


# In[25]:


rfc=RandomForestClassifier()


# In[26]:


model=rfc.fit(X_train,y_train)


# In[27]:


prediction=model.predict(X_test)


# In[28]:


from sklearn.metrics import accuracy_score


# In[29]:


accuracy_score(y_test,prediction)


# Model 2:

# In[30]:


from sklearn.linear_model import LogisticRegression


# In[31]:


X1=data.drop(['Class'],axis=1)


# In[32]:


y1=data['Class']


# In[33]:


X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.3,random_state=123)


# In[34]:


lr=LogisticRegression()


# In[36]:


model2=lr.fit(X1_train,y1_train)


# In[37]:


prediction2=model2.predict(X1_test)


# In[38]:


accuracy_score(y1_test,prediction2)


# Model 3:

# In[39]:


from sklearn.tree import DecisionTreeRegressor


# In[40]:


X2=data.drop(['Class'],axis=1)


# In[41]:


y2=data['Class']


# In[42]:


dt=DecisionTreeRegressor()


# In[43]:


X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.3,random_state=123)


# In[44]:


model3=dt.fit(X2_train,y2_train)


# In[45]:


prediction3=model3.predict(X2_test)


# In[46]:


accuracy_score(y2_test,prediction3)


# All of our models performed with a very high accuracy.

# In[ ]:




