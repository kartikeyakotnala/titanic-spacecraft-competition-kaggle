#!/usr/bin/env python
# coding: utf-8

# In[956]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,Normalizer,StandardScaler,RobustScaler,normalize,MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import brier_score_loss
from sklearn.cluster import KMeans


# In[924]:


df=pd.read_csv(r'C:\Users\karti\OneDrive\Desktop\titanic\train.csv')
model=LogisticRegression()
Smodel=SVC(C=1,kernel='poly',degree=2)
df.head()


# In[925]:


df.drop(['HomePlanet','Cabin','Destination','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Name'],axis='columns',inplace=True)


# In[926]:


df


# In[927]:


#sns.barplot(x=df.CryoSleep[df.CryoSleep==0],y=df.CryoSleep[df.CryoSleep==1],hue=df.Transported)


# In[ ]:





# In[928]:


df.columns[df.isna().any()]


# In[929]:


df['VIP'].fillna(df['VIP'].median(),inplace=True)
df['Age'].fillna(df['Age'].median(),inplace=True)
df['CryoSleep'].fillna(df['CryoSleep'].median(),inplace=True)


# In[930]:


le=LabelEncoder()


# In[931]:


df.CryoSleep=le.fit_transform(df.CryoSleep)
df.VIP=le.fit_transform(df.VIP)


# In[932]:


df.CryoSleep=pd.to_numeric(df.CryoSleep)
df.VIP=pd.to_numeric(df.VIP)
df.Age=pd.to_numeric(df.Age)


# In[933]:


X=df.drop(['PassengerId','Transported'],axis='columns')
X


# In[934]:


#scaler = MinMaxScaler()
#scaler_df = scaler.fit_transform(X)
#scaler_df = pd.DataFrame(scaler_df, columns =['CryoSleep','Age','VIP'])
#X=scaler_df


# In[935]:


y=df.Transported


# In[936]:


X


# In[937]:


X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)


# In[938]:


clf=GridSearchCV(LogisticRegression(),{
    'penalty' : ['l2'],
    'class_weight':[{0:1.2,1:0.1,1:1.95}],
    'C':[.1]
}
,cv=4)


# In[ ]:





# In[939]:


clf.fit(X,y)
brier_score_loss(y_test,clf.predict(X_test))


# In[940]:


clf.score(X,y)


# In[941]:


clf.predict_proba(X)[:,1]


# In[942]:


X


# In[943]:


pd.DataFrame(clf.cv_results_)


# In[944]:


#for testing

df_test=pd.read_csv(r"C:\Users\karti\OneDrive\Desktop\titanic\test.csv")


# In[945]:


df_test.drop(['HomePlanet','Cabin','Destination','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Name'],axis='columns',inplace=True)
df_test['VIP'].fillna(df_test['VIP'].median())
df_test['Age'].fillna(df_test['Age'].mean(),inplace=True)
df_test['CryoSleep'].fillna(df_test['CryoSleep'].median())
df_test.CryoSleep=le.fit_transform(df_test.CryoSleep)
df_test.VIP=le.fit_transform(df_test.VIP)
X_test=df_test.drop(['PassengerId'],axis='columns')


# In[946]:


#scaler = MinMaxScaler()
#scaler_df = scaler.fit_transform(X_test)
#scaler_df = pd.DataFrame(scaler_df, columns =['CryoSleep','Age','VIP'])
#X_test=scaler_df


# In[947]:


y_test=clf.predict(X_test)


# In[948]:


y_test=clf.predict_proba(X_test)[:,1]


# In[949]:


df_test['Transported']=pd.DataFrame(y_test, columns=['Transported'])
#df_test.loc[df_test['Transported']>0.89,'Transported']=1
#df_test.loc[df_test['Transported']<0.3,'Transported']=0


# In[953]:


result=pd.concat([df_test['PassengerId'],df_test['Transported']],axis='columns')


# In[954]:


result.to_csv(r'C:\Users\karti\OneDrive\Desktop\titanic\ResultsKneighbourprob1.csv')


# In[955]:


result


# In[ ]:





# In[ ]:




