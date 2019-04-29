
# coding: utf-8

# In[1]:


import pandas as pd
import sys
import os
import numpy as np
from scipy.stats import norm
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
#from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Reading the csv files
print("Please provide the first dataset path:")
df1 = pd.read_csv(r"C:\Users\ashu1\OneDrive\Desktop\dc-residential-properties\Book1.csv")
df2 = pd.read_csv(r"C:\Users\ashu1\OneDrive\Desktop\dc-residential-properties\Book2.csv")
print(df1.shape)
print(df2.shape)


# In[3]:


list1 = list(df1)                               # Getting all column names into list
list2 = list(df2)
match_col = set(list1).intersection(list2)      # Go get common column from two datasets
match_col = list(match_col)                     # Converting set into list
df = pd.merge(df1, df2, right_index = True, left_index = True)# merging two datsets on the basis of common column
df = df[df.columns.drop(list(df.filter(regex = '_y')))]


# In[4]:


Target = df.iloc[:,-1]                          # Getting the last column as a Target from the dataset  
df.drop(df.columns[[-1,]], axis=1, inplace=True)# Dropping the last column from the datset


# In[5]:


df.isnull().values.any()                        # checking for null values


# In[6]:


df = pd.get_dummies(df)                         # dummy encoding, converting categorical values into continous
feat_labels = list(df)
df.shape
# getting names of columns of the dataset and saving it as a list


# In[7]:


from sklearn.preprocessing import StandardScaler     # import StandardScaler to get data into one scale

sc = StandardScaler()
df = sc.fit_transform(df)                           


# In[8]:


# Regressor for regression task
def Regression (df, Target):              
    regr = RandomForestRegressor(n_jobs = -1, n_estimators = 10, random_state = 0)
    model = regr.fit(df, Target)
    print(mean_squared_error(Target, regr.predict(df)))

    for feature in zip(feat_labels, regr.feature_importances_):
        print(feature)
    return model


# In[9]:


def Classification (df, Target):
    clf = RandomForestClassifier(n_jobs = -1, n_estimators = 10, random_state = 0)
    model = clf.fit(df, Target)
    print(accuracy_score(Target, clf.predict(df)))
    
    for feature in zip(feat_labels, clf.feature_importances_):
        print(feature)
    return model


# In[10]:


# Choose between classifier and regression task with the help of datatype of target variable
if Target.dtype == 'object':
    model = Classification(df, Target)
    #a = clf
else:
    Target = np.log1p(Target)            # if its regression then get logarithmic values of target variable
    model = Regression(df, Target)


# In[11]:


# Create a selector object that will use the random forest classifier to identify
# features that have an importance of more than 0.05
sfm = SelectFromModel(model, threshold = 0.05)

# Train the selector
sfm.fit(df, Target)


# In[12]:


# Print the names of the most important features
Imp_Labels = []
for feature_list_index in sfm.get_support(indices = True):
    print(feat_labels[feature_list_index])
    Imp_Labels.append(feat_labels[feature_list_index])


# In[13]:


# Transform the data to create a new dataset containing only the most important features
df_imp = sfm.transform(df)
New_df = pd.DataFrame(df_imp)
New_df.columns = Imp_Labels


# In[14]:


New_df.shape


# In[15]:


New_df.head(5)


# In[16]:


# clf = RandomForestClassifier(n_jobs = -1, n_estimators = 10, random_state = 0)
# model = clf.fit(New_df, Target)
# print(accuracy_score(Target, clf.predict(New_df)))

