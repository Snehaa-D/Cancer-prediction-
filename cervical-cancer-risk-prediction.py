#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile


# In[2]:


cancer_df=pd.read_csv('C:/Users/sneha/OneDrive/Documents/Minor Project 7sem/kag_risk_factors_cervical_cancer.csv')
cancer_df
cancer_df.tail(20)


# In[3]:


#dataset info
cancer_df.info()


# In[4]:


#statistical info
cancer_df.describe()


# In[5]:


cancer_df=cancer_df.replace('?',np.nan)
cancer_df


# In[6]:


cancer_df.isnull()


# In[7]:


plt.figure(figsize=(30,30))
sns.heatmap(cancer_df.isnull())


# In[8]:


cancer_df=cancer_df.drop(columns=['STDs: Time since first diagnosis','STDs: Time since last diagnosis'])
cancer_df


# In[9]:


cancer_df=cancer_df.apply(pd.to_numeric)
cancer_df.info()


# In[25]:


cancer_df.describe()
cancer_df.mean()


# In[11]:


cancer_df=cancer_df.fillna(cancer_df.mean())
cancer_df


# In[12]:


sns.heatmap(cancer_df.isnull())


# In[13]:


cancer_df['Age'].min()
cancer_df['Age'].max()
cancer_df[cancer_df['Age']==84]


# In[14]:


corr_matrix=cancer_df.corr()
corr_matrix


# In[15]:


plt.figure(figsize=(30,30))
sns.heatmap(corr_matrix)
plt.show()


# In[16]:


cancer_df.hist(figsize=(20,20),bins=12)


# In[17]:


target_df=cancer_df['Biopsy']
input_df=cancer_df.drop(columns=['Biopsy'])


# In[18]:


input_df.shape


# In[19]:


X=np.array(input_df).astype('float32')
Y=np.array(target_df).astype('float32')
Y.shape


# In[20]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler=StandardScaler()
X=scaler.fit_transform(X)


# In[21]:


X


# In[38]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
X_test,X_val,Y_test,Y_val=train_test_split(X_test,Y_test,test_size=0.5)


# In[39]:


pip install xgboost


# In[40]:


#train
import xgboost as xgb
model=xgb.XGBClassifier(learning_rate=0.1,max_depth=5,n_estimators=10)
model.fit(X_train,Y_train)


# In[41]:


result_train=model.score(X_train,Y_train)
result_train


# In[43]:


#predict score of trained model using testing dataset
result_test=model.score(X_test,Y_test)
result_test


# In[45]:


#make prediciton on test data
y_predict=model.predict(X_test)


# In[47]:


from sklearn.metrics import confusion_matrix


# In[48]:


cm=confusion_matrix(y_predict,Y_test)
sns.heatmap(cm,annot=True)


# In[ ]:




