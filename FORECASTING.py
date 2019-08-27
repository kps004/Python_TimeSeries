#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as datetime
import xlsxwriter
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dd = pd.read_csv("powernetworks.csv")


# In[ ]:





# In[3]:


#dd = dd[:80000]
count = dd["LCLid"].count()
mac = dd.LCLid.unique()
dd['KWh'] = dd['KWh'].astype(float)
macid = []
i=0
for x in mac:
    macid.append(i) 
    i=i+1  
i=int(i)
dd["time"]=0
i


# In[12]:


t= pd.timedelta_range(start='1 days',periods = count , freq='H')
time = pd.Series(range(len(t)), index=t)

dd['time'] = time
dd["time"] = t
#dd['LCLid'] = df.LCLid
#dd['KWh'] = df.KWh
#dd['time'] = pd.to_datetime(dd["time"])


# In[14]:



dd['hours'] = t
dd['date'] = 0
#dd['DateTime'] = pd.to_datetime(df['DateTime'])
#dd['date'], df['hours'] = df['time'].dt.normalize(), df['time'].dt.time
dd


# In[15]:


values = []

for x in range(i):
        ids = dd.groupby(by = "LCLid").get_group(mac[x])
        ids = ids[['KWh','time']]
        values.append(ids)
        macid.append(ids) 
house = values[2]


# In[16]:


house.plot()


# In[17]:


from statsmodels.graphics.tsaplots import plot_acf
plot_acf(house["KWh"])


# In[18]:


x = house["KWh"]
x = np.array(x).reshape((-1, 1))
y = house['time']
y= np.array(y).reshape((-1, 1))


# In[19]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[20]:


model = LinearRegression()


# In[21]:


model.fit(x,y)


# In[23]:


y_predicted = model.predict(x)
y_predicted


# In[ ]:


rmse = mean_squared_error(y, y_predicted)
r2 = r2_score(y, y_predicted)

# printing values
print('Slope:' ,regression_model.coef_)
print('Intercept:', regression_model.intercept_)
print('Root mean squared error: ', rmse)
print('R2 score: ', r2)

# plotting values

# data points
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')

# predicted values
plt.plot(x, y_predicted, color='r')
plt.show()

