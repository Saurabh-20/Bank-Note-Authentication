#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


df=pd.read_csv(r"C:\Users\pc\Downloads\BankNote_Authentication.csv")


# In[4]:


df


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


df.columns


# In[8]:


D=df.drop(['variance', 'skewness', 'curtosis', 'entropy'],axis=1)


# In[9]:


D


# In[10]:


D=D.values


# In[11]:


len(D)


# In[13]:


I=df.drop(['class'],axis=1)


# In[14]:


I


# In[15]:


I=I.values


# In[16]:


I_train,I_test,D_train,D_test=train_test_split(I,D,test_size=0.2,random_state=0)


# In[17]:


I_train


# In[18]:


I_test


# In[19]:


D_train


# In[20]:


D_test


# In[21]:


from sklearn.svm import SVC
s=SVC(kernel='linear')
s.fit(I_train,D_train)


# In[22]:


p=s.predict(I_test)


# In[23]:


df1=pd.DataFrame({'ACTUAL':D_test.flatten(),'PREDICTED':p})
df1


# In[24]:


from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(D_test,p)


# In[25]:


accuracy_score(D_test,p)


# In[26]:


x=df1.ACTUAL
y=df1.PREDICTED
import matplotlib.pyplot as plt
plt.scatter(np.arange(275),x)
plt.scatter(np.arange(275),y)


# In[ ]:




