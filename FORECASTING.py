
#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble.forest import RandomForestRegressor

# In[2]:


df = pd.read_csv("Book2.csv")


# In[3]:



count = df["LCLid"].count()


# In[4]:


li = [1]
for x in range(count):
     li.append(x)
   


# In[5]:


colu = {"key": li}
dd = pd.DataFrame(colu)

df["index"] = dd.copy()
lclib  = df.groupby(by = "LCLid")
A = lclib.get_group("MAC000002")


# In[6]:


lclid = df.groupby(by = "LCLid")
a = lclid.get_group("MAC000002")
c = a["KWh"].count()
test_value = int(c - (0.9*c)) 
a = a[["KWh","index"]]


# In[ ]:





# In[10]:


X = a['KWh']
y = a['index']
    
X_train = X[X.index < 35 ]
y_train = y[y.index < 35 ]              
    
X_test = X[X.index >= 35 ]    
y_test = y[y.index >= 35  ]
X


# In[19]:


# build our RF model
RF_Model = RandomForestRegressor(n_estimators=1000,
                                 max_features=1, oob_score=True)

# let's get the labels and features in order to run our 
# model fitting
labels = y_train#[:, None]
features = X_train[:, None]

# Fit the RF model with features and labels.
rgr=RF_Model.fit(features, labels)

# Now that we've run our models and fit it, let's create
# dataframes to look at the results
X_test_predict=pd.DataFrame(
    rgr.predict(X_test[:, None])).rename(
    columns={0:'KWh'}).set_index('KWh')
X_train_predict=pd.DataFrame(
    rgr.predict(X_train[:, None])).rename(
    columns={0:'KWh'}).set_index('KWh')

# combine the training and testing dataframes to visualize
# and compare.
RF_predict = X_train_predict.append(X_test_predict)


# In[17]:


a[['KWh','index']].plot()


# In[ ]:




