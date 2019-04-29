
# coding: utf-8

# In[91]:


import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
#%matplotlib inline


# In[92]:


df1 = pd.read_csv(r"C:\Users\ashu1\OneDrive\Desktop\DC_Housing_CLEAN_5000.csv")
df2 = pd.read_csv(r"C:\Users\ashu1\OneDrive\Desktop\DC_CRIME_CLEAN_5000.csv")
print(df1.shape)
print(df2.shape)


# In[93]:


print(df1.isnull().values.any())
print(df2.isnull().values.any())


# In[94]:


Target1 = df1.iloc[:,-1]
df1.drop(df1.columns[[-1,]], axis = 1, inplace = True)
Target2 = df2.iloc[:,-1]
df2.drop(df2.columns[[-1,]], axis =1, inplace = True)
print(Target1.dtype)
print(Target2.dtype)


# In[95]:


df1 = pd.get_dummies(df1)                         # dummy encoding, converting categorical values into continous
feat_labels1 = list(df1)
df2 = pd.get_dummies(df2)                         # dummy encoding, converting categorical values into continous
feat_labels2 = list(df2)


# In[96]:


from sklearn.preprocessing import StandardScaler     # import StandardScaler to get data into one scale

sc = StandardScaler()
df1 = sc.fit_transform(df1) 
df2 = sc.fit_transform(df2)


# In[97]:


def Regression(df, Target):
    regr = RandomForestRegressor(n_jobs = -1, n_estimators = 10, random_state = 42)
    model = regr.fit(df, Target)
    print(mean_squared_error(Target, regr.predict(df)))

#     for feature in zip(feat_labels, regr.feature_importances_):
#         print(feature)
    return model


# In[98]:


def Classification(df, Target):
    clf = RandomForestClassifier(n_jobs = -1, n_estimators = 10, random_state = 42 )
#     clf = RandomForestClassifier(random_state = 42)
#     clf = GridSearchCV(estimator = clf, param_grid = param_grid_clf, cv = 5)
    model = clf.fit(df, Target)
    print(accuracy_score(Target, clf.predict(df)))
    
#     for feature in zip(feat_labels, clf.feature_importances_):
#         print(feature)
    return model


# In[99]:


print(Target1.dtype)
print(Target2.dtype)


# In[100]:


if (Target1.dtype == 'object') and (Target2.dtype == 'int64' or 'float64'):
    Target2 = np.log1p(Target2)
    model1 = Classification(df1, Target1)
    model2 = Regression(df2, Target2)

elif Target1.dtype == 'object' and Target2.dtype == 'object':
    model1 = Classification(df1, Target1)
    model2 = Classification(df2, Target2)

    
elif (Target1.dtype == 'int64' or 'float64') and (Target2.dtype == 'object'):
    Target1 = np.log1p(Target1)
    model1 = Regression(df1, Target1)
    model2 = Classification(df2, Target2)
    
else:
    Target1 = np.log1p(Target1)
    Target2 = np.log1p(Target2)
    model1 = Regression(df1, Target1)
    model2 = Regression(df2, Target2)


# In[101]:


sfm = SelectFromModel(model1, threshold = 0.05)
sfm.fit(df1, Target1)

Imp_Labels1 = []
for feature_list_index in sfm.get_support(indices = True):
    print(feat_labels1[feature_list_index])
    Imp_Labels1.append(feat_labels1[feature_list_index])

df_imp1 = sfm.transform(df1)
New_df1 = pd.DataFrame(df_imp1)
New_df1.columns = Imp_Labels1


# In[102]:


sfm = SelectFromModel(model2, threshold = 0.05)
sfm.fit(df2, Target2)

Imp_Labels2 = []
for feature_list_index in sfm.get_support(indices = True):
    print(feat_labels2[feature_list_index])
    Imp_Labels2.append(feat_labels2[feature_list_index])
    
df_imp2 = sfm.transform(df2)
New_df2 = pd.DataFrame(df_imp2)    
New_df2.columns = Imp_Labels2


# In[103]:


result = pd.concat([New_df1, New_df2], axis = 1, ignore_index = False)


# In[104]:


result.shape


# In[105]:


result.head(5)


# In[106]:


# clf = RandomForestClassifier(n_jobs = -1, n_estimators = 10, random_state = 0)
# model = clf.fit(result, Target2)
# print(accuracy_score(Target2, clf.predict(result)))

