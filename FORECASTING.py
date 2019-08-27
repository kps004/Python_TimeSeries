#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as datetime
import xlsxwriter
get_ipython().run_line_magic('matplotlib', 'inline')


dd = pd.read_csv("powernetworks.csv")


#dd = dd[:80000]
#count total data
count = dd["LCLid"].count()
mac = dd.LCLid.unique()
dd['KWh'] = dd['KWh'].astype(float)
macid = []
i=0

# count unique id in the list
for x in mac:
    macid.append(i) 
    i=i+1  
i=int(i)


dd["time"]=0
# create data for date time into time column
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
values = [] #create list for grouping data

for x in range(i):
        ids = dd.groupby(by = "LCLid").get_group(mac[x])
        ids = ids[['KWh','time']]
        values.append(ids) 
        #initialize all the uniques id aas a single group data
        macid.append(ids) 
house = values[2]



house.plot()
#plot graph for data
# import model for auto correlation
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(house["KWh"])
# convert data for 2d for fit model


x = house["KWh"]
x = np.array(x).reshape((-1, 1))
y = house['time']
y= np.array(y).reshape((-1, 1))

#import sckit lib for Linear Regression Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
model = LinearRegression()
model.fit(x,y)

y_predicted = model.predict(x)
y_predicted


# find root square values
#Error values
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

