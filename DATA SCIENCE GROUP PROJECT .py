#!/usr/bin/env python
# coding: utf-8

# # DATA SCIENCE PROJECT.

# # TOPIC :- DELHI HOUSE PRICES PREDICTION.

# * Name:-Ronak Saraswat
# * course:-B.Tech
# * branch:-Computer Science

# In[2]:


import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import numpy as np
import seaborn as sns


# * 1)-Getting dataset.
# * 2)-Importing libraries.
# * 3)-read dataset.
# * 4)-data processing.
# * 5)-clean the dataset.
# * 6)-identifying and handling missing values.
# * 7)-encoding the categorical values.
# * 8)-splitting the dataset.
# * 9)-feature scaling.

# In[3]:


# here we are reading our dataset name-"delhi.csv" by using pandas library.
df=pd.read_csv("delhi.csv")
df


# In[4]:


df.describe()


# In[5]:


df.info()


# In[7]:


df.shape


# In[9]:


df.head()


# In[10]:


print(df['Status'].value_counts())
print("_________________***********************__________________________________________________")
print(df['Furnished_status'].value_counts())
print("_________________***********************__________________________________________________")
print(df['neworold'].value_counts())
print("_________________***********************__________________________________________________")
print(df['type_of_building'].value_counts())
print("_________________***********************__________________________________________________")
print(df['Landmarks'].value_counts())
print("_________________***********************__________________________________________________")
print(df['Lift'].value_counts())
print("_________________***********************__________________________________________________")
print(df['Balcony'].value_counts())
print("_________________***********************__________________________________________________")
print(df['parking'].value_counts())


# In[11]:


df.isna().sum()


# In[12]:


df.isna().sum().sum()


# In[14]:


dic={'Ready to Move':1,'Under Construction':2}
df['Status']=df['Status'].map(dic)
df.head()


# In[15]:


dic={'Semi-Furnished':1,'Unfurnished':2,'Furshied':3}
df['Furnished_status']=df['Furnished_status'].map(dic)
df.head()


# In[16]:


dic={'New Property':1,'Resale':2}
df['neworold']=df['neworold'].map(dic)
df.head()


# In[19]:


dic={'Flat':1,'Individual House':2}
df['type_of_building']=df['type_of_building'].map(dic)
df.head()


# In[26]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df.Address=le.fit_transform(df.Address)
df.Landmarks=le.fit_transform(df.Landmarks)
df.desc=le.fit_transform(df.desc)
df


# In[27]:


df1=df.fillna(value=0)
df1


# In[30]:


df1.isna().sum()


# In[58]:


corr=df1.corr()
corr


# # Linear regression.

# In[33]:


x=df1[['area','latitude','longitude','Bedrooms','Bathrooms','Balcony','Status','Furnished_status','neworold','parking','Lift','type_of_building','Price_sqft','desc','Address','Landmarks']]
y=df1['price']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=123)
print(x.shape)
print(x_train.shape)
print(x_test.shape)


# In[39]:


from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_train,y_train)
y_pred=lm.predict(x_test)
print(lm.score(x_test,y_test))


# In[38]:


coeff=pd.DataFrame(lm.coef_,x.columns,columns=['coefficient'])
coeff


# In[35]:


c=lm.coef_
i=lm.intercept_
print(c,i)


# In[54]:


print(f"Accuracy of Test Data is {round(lm.score(x_test, y_test)*100,2)}%")
print(f"Accuracy of Training Data is {round(lm.score(x_train, y_train)*100,2)}%")


# In[43]:


from sklearn.metrics import r2_score
r2score=r2_score(y_test,y_pred)
print(r2score)


# In[44]:


prediction=lm.predict(x_test)
plt.scatter(y_test,prediction)


# In[56]:


y_test


# In[57]:


y_pred


# # Ridge Regression.

# In[67]:


from sklearn.linear_model import Ridge, Lasso
rd=Ridge()
rd.fit(x_train,y_train)
print(rd.score(x_test,y_test))
y_pred=lm.predict(x_test)
print(r2_score(y_test,y_pred))


# # Lasso Regression

# In[68]:


ls=Lasso()
ls.fit(x_train,y_train)
print(ls.score(x_test,y_test))
y_pred=lm.predict(x_test)
print(r2_score(y_test,y_pred))


# # Decision Tree

# In[70]:


from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
preds = dt.predict(x_test)
print(r2_score(y_test, preds))
print(dt.score(x_test,y_test))


# # Random Forest Regression

# In[69]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
preds = rf_model.predict(x_test)
print('Random Forest:', r2_score(y_test, preds))


# # K Nearest Neighbor

# In[75]:


from sklearn.neighbors import KNeighborsRegressor
regressor=KNeighborsRegressor(n_neighbors=5)
regressor.fit(x_train,y_train)
regressor.score(x_test,y_test)


# In[76]:


x_test.iloc[-1,:]


# In[77]:


regressor.predict([x_test.iloc[-1,:]])


# In[78]:


y_test.iloc[-1]


# # The best model is decision Tree...

# In[ ]:




