#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[2]:





# In[2]:


import pandas as pd
import numpy as np


# In[5]:


df=pd.read_csv(r"C:\Users\pc\Downloads\bill_authentication.csv")


# In[4]:


from sklearn.model_selection import train_test_split


# In[6]:


df.columns


# In[9]:


D=df.drop(['Variance', 'Skewness', 'Curtosis', 'Entropy'],axis=1)


# In[12]:


D
D=D.values


# In[15]:


len(D)


# In[16]:


D


# In[17]:


I=df.drop(['Class'],axis=1)


# In[18]:


I


# In[19]:


I=I.values


# In[20]:


I


# In[29]:


I_train,I_test,D_train,D_test=train_test_split(I,D,test_size=0.2,random_state=0)


# In[32]:


I_train


# In[33]:


I_test


# In[37]:


D_train


# In[35]:


D_test


# In[38]:


from sklearn.svm import SVC
s=SVC(kernel='linear')
s.fit(I_train,D_train)


# In[40]:


p=s.predict(I_test)


# In[42]:


df1=pd.DataFrame({'ACTUAL':D_test.flatten(),'PREDICTED':p})
df1


# In[43]:


df1.to_csv(r"C:\Users\pc\Downloads\currency.csv")


# In[44]:


from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(D_test,p)


# In[45]:


accuracy_score(D_test,p)


# In[47]:


a=np.array([[0.60911,3.9334,-3.782,-4.4863],[0.20121,1.7892,-2.790,-3.6789]])


# In[48]:


s.predict(a)


# In[61]:


x=df1.ACTUAL
y=df1.PREDICTED
import matplotlib.pyplot as plt
plt.scatter(np.arange(275),x)
plt.scatter(np.arange(275),y)


# In[1]:





# In[3]:





# In[6]:





# In[3]:





# In[4]:





# In[5]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




