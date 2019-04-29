
# coding: utf-8

# In[69]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
from pandas.api.types import is_string_dtype


# In[70]:


df1 = pd.read_csv(r"C:\Users\ashu1\OneDrive\Desktop\DC_Housing_CLEAN_5000.csv")
df2 = pd.read_csv(r"C:\Users\ashu1\OneDrive\Desktop\DC_CRIME_CLEAN_5000.csv")
print(df1.shape)
print(df2.shape)


# In[71]:


df = pd.concat([df1, df2], axis = 1)
df.isnull().values.any()


# In[72]:


Target = df.iloc[:,-1]
df.drop(df.columns[[-1,]], axis=1, inplace=True)
df_con = df.select_dtypes(include=['float64', 'int64'])
df_cat = df.select_dtypes(include=['object'])


# In[73]:


# def correlation(df_con, threshold):
#     col_corr = set() # Set of all the names of deleted columns
#     corr_matrix = df_con.corr()
#     for i in range(len(corr_matrix.columns)):
#         for j in range(i):
#             if corr_matrix.iloc[i, j] >= threshold:
#                 colname = corr_matrix.columns[i] # getting the name of column
#                 col_corr.add(colname)
#                 if colname in df_con.columns:
#                     del df_con[colname] # deleting the column from the dataset
#     return df_con


# In[74]:


#correlation(df_con, 0.7)
#df_con[df_con.columns[0]]


# In[75]:


# if is_string_dtype(df_con[df_con.columns[0]]) == True:
#     print("Object")
# else:
#     print("Int")


# In[76]:


# df_cat = pd.get_dummies(df_cat)


# In[77]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[78]:


# df_con = sc.fit_transform(df_con)
# df_cat = sc.fit_transform(df_cat)


# In[79]:


def Regression_con(df, Target):
    df = sc.fit_transform(df)
    regr = RandomForestRegressor(n_jobs = -1, n_estimators = 100, random_state = 0)
    model = regr.fit(df, Target)
    print(mean_squared_error(Target, regr.predict(df)))
    return model


# In[80]:


def Regression_cat(df, Target):
    #df = pd.get_dummies(df)
    df = sc.fit_transform(df) 
    regr = RandomForestRegressor(n_jobs = -1, n_estimators = 100, random_state = 0)
    model = regr.fit(df, Target)
    print(mean_squared_error(Target, regr.predict(df)))
    return model


# In[81]:


def Classification_con(df, Target):
    df = sc.fit_transform(df)
    clf = RandomForestClassifier(n_jobs = -1, n_estimators = 100, random_state = 10 )
    model = clf.fit(df, Target)
    print(accuracy_score(Target, clf.predict(df)))
    return model


# In[82]:


def Classification_cat(df, Target):
    #df = pd.get_dummies(df)
    df = sc.fit_transform(df)
    clf = RandomForestClassifier(n_jobs = -1, n_estimators = 100, random_state = 10 )
    model = clf.fit(df, Target)
    print(accuracy_score(Target, clf.predict(df)))
    return model


# In[83]:


if Target.dtype == 'object':
    df_cat = pd.get_dummies(df_cat)
    #df_con = sc.fit_transform(df_con)
    #df_cat = sc.fit_transform(df_cat)
    feat_labels_con = list(df_con)
    feat_labels_cat = list(df_cat)
    model_con = Classification_con(df_con, Target)
    model_cat = Classification_cat(df_cat, Target)
else:
    df_cat = pd.get_dummies(df_cat)
    #df_con = sc.fit_transform(df_con)
    #df_cat = sc.fit_transform(df_cat)
    feat_labels_con = list(df_con)
    feat_labels_cat = list(df_cat)
    Target = np.log1p(Target)
    model_con = Regression_con(df_con, Target)
    model_cat = Regression_cat(df_cat, Target)


# In[84]:


sfm = SelectFromModel(model_con, threshold = 0.05)
sfm.fit(df_con, Target)

Imp_Labels_con = []
for feature_list_index in sfm.get_support(indices = True):
    print(feat_labels_con[feature_list_index])
    Imp_Labels_con.append(feat_labels_con[feature_list_index])

df_imp_con = sfm.transform(df_con)
New_df_con = pd.DataFrame(df_imp_con)
New_df_con.columns = Imp_Labels_con


# In[85]:


sfm = SelectFromModel(model_cat, threshold = 0.05)
sfm.fit(df_cat, Target)

Imp_Labels_cat = []
for feature_list_index in sfm.get_support(indices = True):
    print(feat_labels_cat[feature_list_index])
    Imp_Labels_cat.append(feat_labels_cat[feature_list_index])

df_imp_cat = sfm.transform(df_cat)
New_df_cat = pd.DataFrame(df_imp_cat)
New_df_cat.columns = Imp_Labels_cat


# In[86]:


result = pd.concat([New_df_con, New_df_cat], axis = 1, ignore_index = False)


# In[87]:


result.shape


# In[88]:


result.head(5)

