#!/usr/bin/env python
# coding: utf-8

# In[3]:


from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


# In[4]:


app=pd.read_csv('C:\Program Files\PYTHON TEXT\googleplaystore.csv')


# In[ ]:





# In[6]:


app.info()


# In[87]:


app.head()


# In[9]:


app.shape


# In[10]:


app.isnull().sum()


# In[11]:


app.dropna(inplace= True)
app.isnull().sum()
app.shape


# In[12]:


app.Size.value_counts()


# In[13]:


def Mb_to_kb(size):
     if size.endswith('k'):
        return float(size[:-1])
     elif size.endswith('M'):
        return float(size[:-1])*1000
     else:
        return size


# In[14]:


app['Size'] = app['Size'].replace(['Varies with device'],'Nan')
app['Size'] = app['Size'].apply(lambda x: Mb_to_kb(x))
app['Size'] = app['Size'].astype(float)
app['Size'].fillna(app.groupby('Category')['Size'].transform('mean'),inplace = True)


# In[15]:


app['Reviews']=app["Reviews"].astype(int)


# In[16]:


app['Installs']=app['Installs'].apply(lambda x:x.replace(",",''))
app['Installs']=app['Installs'].apply(lambda x:x.replace("+",''))
app['Installs']=app['Installs'].astype(int)


# In[17]:


app['Price']=app['Price'].apply(lambda x:x.replace("$",''))
app['Price']=app['Price'].astype(float)


# In[18]:


len(app[(app['Rating'] < 1) & (app['Rating'] > 5)])


# In[19]:


app.shape


# In[20]:


len(app[app.Installs<app.Reviews])


# In[21]:


i3=app[app.Installs<app.Reviews].Installs.index


# In[22]:


app.drop(axis=0 ,index=i3, inplace=True)


# In[23]:


app.shape


# In[24]:


len(app[(app['Type'] == 'Free') & (app['Price'] != 0)])


# In[25]:


plt.figure(figsize=(16, 6))
b=sns.boxplot(app.Price)
b.axes.set_title("Price Boxplot",fontsize=30)
b.set_xlabel("Price(USD)",fontsize=20)
b.tick_params(labelsize=15)


# In[26]:


plt.figure(figsize=(16, 6))
b=sns.boxplot(app.Reviews)
b.axes.set_title("Reviews",fontsize=30)
b.set_xlabel("Number of Reviews",fontsize=20)
b.tick_params(labelsize=15)


# In[27]:


plt.figure(figsize=(16, 6))
h=sns.histplot(app.Rating,bins=60,kde=True)
h.axes.set_title("Rating",fontsize=30)
h.set_xlabel("Rating",fontsize=20)
h.set_ylabel("Count",fontsize=20)
h.tick_params(labelsize=15)


# In[28]:


plt.figure(figsize=(16, 6))
h=sns.histplot(app.Size,bins=50,kde=True)
h.axes.set_title("Size of the App",fontsize=30)
h.set_xlabel("Size(Kb)",fontsize=20)
h.set_ylabel("Count",fontsize=20)
h.tick_params(labelsize=15)


# In[29]:


app.Price.quantile([0.1,0.25,0.50,0.75,0.90,0.95,0.99])


# In[30]:


len(app[app.Price >=200])


# In[31]:


app.Price.mean()


# In[32]:


i7=app[app.Price >=200].Price.index


# In[33]:


app.drop(axis=0 ,index=i7, inplace=True)


# In[34]:


app.shape


# In[35]:


len(app[app.Reviews >=2000000])


# In[36]:


i8=app[app.Reviews >=2000000].Reviews.index


# In[37]:


app.drop(axis=0 ,index=i8, inplace=True)


# In[38]:


app.shape


# In[39]:


app.Installs.quantile([0.1,0.25,0.50,0.75,0.90,0.95,0.99])


# In[40]:


min_threshold,max_threshold=app.Installs.quantile([0.05,0.95])
inp=app[(app.Installs>min_threshold) & (app.Installs<max_threshold)]


# In[41]:


inp.shape


# In[42]:


plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
j=sns.jointplot(y = "Rating", x = "Price", data = inp,height=8)
j.fig.suptitle("Rating vs Price",fontsize=30)
j.ax_joint.set_xlabel('Price(USD)',fontsize=20)
j.ax_joint.set_ylabel('Rating',fontsize=20)


# In[43]:


plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
j=sns.jointplot(y = "Rating", x = "Size", data = inp,height=8)
j.fig.suptitle("Rating vs Size",fontsize=30)
j.ax_joint.set_xlabel('Size(Kb)',fontsize=20)
j.ax_joint.set_ylabel('Rating',fontsize=20)


# In[44]:


plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
j=sns.jointplot(y = "Rating", x = "Reviews", data = inp,height=8)
j.fig.suptitle("Rating vs Reviews",fontsize=30)
j.ax_joint.set_xlabel('Reviews',fontsize=20)
j.ax_joint.set_ylabel('Rating',fontsize=20)


# In[45]:


plt.figure(figsize=(16,6))
h=sns.boxplot( x='Content Rating',y='Rating',data=inp)
h.axes.set_title("Rating vs Content Rating",fontsize=30)
h.set_xlabel("Content Rating",fontsize=20)
h.set_ylabel("Rating",fontsize=20)
h.tick_params(labelsize=13)


# In[46]:


plt.figure(figsize=(16,6))
h=sns.boxplot( x='Category',y='Rating',data=inp)
h.axes.set_title("Rating vs Category",fontsize=30)
h.set_xlabel("Category",fontsize=20)
h.set_ylabel("Rating",fontsize=20)
h.tick_params(labelsize=13)
plt.xticks(rotation=90);


# In[47]:


groupCat = inp.groupby('Category')
mean_df=groupCat.mean()
mean_df = mean_df.reset_index()
mean_df.sort_values(by='Rating',ascending=False).head(3)


# In[48]:


inp1=inp.copy(deep=True)


# In[49]:


inp1['Installs']= np.log1p(inp1['Installs'])
inp1['Reviews']= np.log1p(inp1['Reviews'])


# In[50]:


fig, axes = plt.subplots(1,2, figsize=(18, 10))
fig.suptitle('Box plot of Installs and Reviews after applying log transformation',fontsize=30)
sns.boxplot( inp1.Installs,ax=axes[0])
sns.boxplot( inp1.Reviews,ax=axes[1],)
axes[0].set_title("Installs",fontsize=20)
axes[1].set_title("Reviews",fontsize=20)


# In[51]:


inp1.drop(['App','Last Updated','Current Ver','Android Ver','Type'],axis=1,inplace=True)


# In[52]:


inp1.shape


# In[53]:


inp1.head()


# In[54]:


cat_cols = ['Category', 'Content Rating','Genres']
inp2 = pd.get_dummies(inp1, columns=cat_cols, drop_first=True)
inp2.head()


# In[55]:


df_train, df_test = train_test_split(inp2, train_size=0.70, random_state=0)
df_train.shape,df_test.shape


# In[56]:


y_train=df_train.Rating
X_train=df_train.drop(['Rating'],axis=1)


# In[57]:


y_test=df_test.Rating
X_test=df_test.drop(['Rating'],axis=1)


# In[58]:


X_train.shape,X_test.shape


# In[59]:


reg =LinearRegression()
reg.fit(X_train, y_train)


# In[60]:


y_pred= reg.predict(X_train)
print('\nR2 on train set: %.2f' % r2_score(y_train, y_pred).round(decimals=2))


# In[61]:


y_pred2=reg.predict(X_test)
r2=r2_score(y_test,y_pred2).round(decimals=2)


# In[62]:


print('\nR2: %.2f' % r2)

print('\nMean Squared Error: %.2f'
      % mean_squared_error(y_test, y_pred2))

print("\nRoot Mean Squared Error",np.sqrt(mean_squared_error(y_test,y_pred2)).round(decimals=2))


# In[63]:


dfReg = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred2,'Difference':y_test-y_pred2})
dfReg


# In[89]:


from mlxtend.evaluate import bias_variance_decomp


# In[90]:


avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(reg, X_train.values,
                                                            y_train.values, X_test.values,
                                                            y_test.values,
                                                            loss='mse',
                                                            num_rounds=50,
                                                            random_seed=20)


# In[91]:


print('Average expected loss: %.3f' % avg_expected_loss)
print('Average bias: %.3f' % avg_bias)
print('Average variance: %.3f' % avg_var)


# In[92]:


#done


# In[ ]:




