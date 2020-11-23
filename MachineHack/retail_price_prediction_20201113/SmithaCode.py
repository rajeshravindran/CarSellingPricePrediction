# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 16:04:12 2020

@author: Smita Shah
"""
### a  machine learning hiring hackathon
import os
root_dir  = os.path.abspath('.')
data_dir  = os.path.join(root_dir, 'data')
train     = os.path.join(data_dir,'Train.csv')
test      = os.path.join(data_dir,'Test.csv' )
submt_fil = os.path.join(data_dir,'Final_submission.csv' )

#
import numpy as np
import pandas as pd
#
import statsmodels.api as sm
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
#
pd.set_option('display.float_format',lambda x:'%.3f'%x)
pd.set_option('display.max_columns',500)
pd.set_option('display.max_colwidth',500)
np.set_printoptions(precision=2)  ##print upto two decimal places
import re
import sys

import warnings
#
if not sys.warnoptions:
    warnings.simplefilter("ignore")
#
import pandas_profiling 
import autoviz
#
try:
    train = pd.read_csv(train)
    print("train dataset has {} samples with {} features each.".format(*train.shape))
    test = pd.read_csv(test.csv)
    print("test dataset has {} samples with {} features each.".format(*test.shape))
    ss= pd.read_csv(submt_fil)
    print("submission dataset has {} samples with {} features each.".format(*ss.shape))
    
except:
    print("Dataset could not be loaded. Is the dataset missing?")
#READ DATA :END======================
#EDA :=====START================
def display_head_tail(df,head_rows,tail_rows):
    display("data head and tail :")
    display(df.head(head_rows).append(df.tail(tail_rows)))
#
display_head_tail(train,2,3)
display_head_tail(test,2,3)

#MISSING VALUE CHECKING AND DATA OVERVIEW
train.info()
print(train.isna().sum())
test.info()
print(test.isna().sum())
train.nunique()
train.shape
train.corr().loc['UnitPrice'] #SEEMS NO  CORRELATION WITH TARGET unitPRICE

#CHECKING FOR -VE QTY
train[train['UnitPrice']<0].head() #unit price is not negative
train[['Quantity','UnitPrice']][train['Quantity']<0].head()
train[train['Quantity']<0].head()

#converting -ve qty to +ve

train['Quantity'][train['Quantity']<0]=train['Quantity'][train['Quantity']<0]*-1
train[train['InvoiceNo']==21750].head()
test[test['Quantity']<0].head()
test['Quantity'][test['Quantity']<0]=test['Quantity'][test['Quantity']<0]*-1

#removing duplicate by keeping first records

duplicate = train[train.duplicated()] 
duplicate.info()
duplicate.head()
train[train['InvoiceNo']== 17505]
duplicate[duplicate['InvoiceNo']== 17505].head()

df_first=train.drop_duplicates(subset=None, keep='first', inplace=False)
df_first.shape
df_first.info()
df_first[df_first['InvoiceNo']== 17505].head()

pr1=df_first.profile_report()
pr1.to_file(output_file="pr1.html")
df_first['StockCode'].value_counts()
df_first['Country'].value_counts()


#boxplot for Quantity  
f, ax = plt.subplots(figsize=(12, 8))
ax = sns.boxplot(data=df_first.loc[:, 'Quantity'])
ax.set(title='Boxplots of Quntity in train dataset')
sns.despine(left=True, bottom=True)


# #pair plot  commented as taking lots of time
# # create specific df that only contains the fields we're interested in
# pairplot_df = df_first.loc[:, ['UnitPrice','StockCode','Country','CustomerID']]

# # create the pairplot
# sns.set(style="dark")
# sns.pairplot(data=pairplot_df)
# plt.show()
# #pair plot  commented as taking lots of time


#plot for qty and unitprice for a country

Country_no = 14
fltr = df_first['Country'] == Country_no
_ = df_first[fltr].set_index('StockCode')[['Quantity','UnitPrice']].plot(figsize=(16, 10), title = f'Country {Country_no}')

#plot for qty and unitprice for a given stock
stock_no = 3249
fltr = df_first['StockCode'] == stock_no
_ = df_first[fltr].set_index('Country')[['Quantity','UnitPrice']].plot(figsize=(16, 10), title = f'Stock {stock_no}')


profile=pandas_profiling.ProfileReport(df_first,minimal=False)
profile.to_file(output_file="pf.html")

print('Checking Data distribution for Train! \n')
col1='UnitPrice'
for col in df_first.columns:
    if col != col1:
        print(f'Distinct entries in {col}: {df_first[col].nunique()}')
        print(f'Common # of {col} entries in test and train: {len(np.intersect1d(df_first[col].unique(), test[col].unique()))}')

import sweetviz as sv
print(dir(sv))
sweet_report = sv.analyze(df_first)
sweet_report.show_html('sweet_report.html')

featcomp=train.columns.to_list()
featcomp.remove('UnitPrice')
df1 = sv.compare(df_first[featcomp], test)
df1.show_html('Compare.html')

#interactive plot 


#import cufflinks as cf
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
#%matplotlib inline
fig=go.Figure(data=[go.Bar(y=df_first)])
plot(fig,auto_open=True)
# cf.go_offline()
# cf.set_config_file(offline=False,world_readable=True)
# df_first[['StockCode','Quantity','UnitPrice']].iplot()

#null value in ascending order 

null_values_per_variable = 100 * (train.isnull().sum()/train.shape[0]).round(3)#.reset_index()
null_values_per_variable.sort_values(ascending=False)


#converting invoice date from obj to date formt
df_first.nunique()
df_first['InvoiceDate'] = pd.to_datetime(df_first['InvoiceDate'])
test['InvoiceDate'] = pd.to_datetime(test['InvoiceDate'])


plt.style.use('fivethirtyeight')
_ = train.set_index('InvoiceDate')['UnitPrice'].plot(figsize=(12, 4))
_ = train.set_index('InvoiceDate')['Country'].plot(figsize=(12, 4))
_ = train.set_index('Country')[['StockCode','Quantity','UnitPrice']].plot(figsize=(12, 4))

#Baseline model
from sklearn.metrics import mean_squared_error
def rmse(y_true, y_pred):
  return mean_squared_error(y_true, y_pred) ** 0.5


from sklearn.metrics import mean_squared_error
def rmse(y_true, y_pred):
  return mean_squared_error(y_true, y_pred) ** 0.5


def download_preds(preds_test, file_name = 'base.csv'):

  ## 1. Setting the target column with our obtained predictions
  ss[TARGET_COL] = preds_test

  ## 2. Saving our predictions to a csv file

  ss.to_csv(file_name, index = False)

  ## 3. Downloading and submitting the csv file
  # from google.colab import files
  # files.download(file_name)



train.groupby(['StockCode'])['UnitPrice'].describe()
df_first.groupby(['StockCode'])['UnitPrice'].describe()

TARGET_COL =  'UnitPrice'

features = [c for c in df_first.columns if c not in [ 'InvoiceNo','Description','CustomerID','UnitPrice']]
print(f'\nThe dataset contains {len(features)} features')
#Target col distribution

_ = sns.distplot(train[TARGET_COL])
_ = plt.title("Target Distribution", fontsize=14)
#stock code wise median unitprice
mapper = train.groupby('StockCode')['UnitPrice'].median()
mapper=mapper.to_frame().reset_index()

#replacing  outliner with median
g=df_first.groupby('StockCode').UnitPrice
df_first['UnitPrice']=(df_first.UnitPrice.where(g.transform('quantile',q=0.85) > df_first.UnitPrice ,g.transform('median')))
g.describe()
df_first.info()
df_first.groupby('StockCode')['UnitPrice'].describe()
#convert invoice date to date and finding year,month,day

dfc = pd.concat([df_first, test], axis=0).reset_index(drop = True)
dfc['InvoiceDate'] = pd.to_datetime(dfc["InvoiceDate"])
dfc.info()
for attr in ['year', 'month', 'day', 'week', 'dayofweek']:
  dfc[attr] = getattr(dfc['InvoiceDate'].dt, attr)
 
df_first, test = dfc[:df_first.shape[0]].reset_index(drop = True), dfc[df_first.shape[0]:].reset_index(drop = True)
df_first.info()  
test.info()
df_first.nunique()

test['InvoiceDate'].max()-test['InvoiceDate'].min()
df_first['InvoiceDate'].max()-df_first['InvoiceDate'].min()

LAST_TRAINING_DAY = df_first['InvoiceDate'].max()
DAYS_TO_VALIDATE = pd.Timedelta(days = 90)
VAL_FIRST_DAY = LAST_TRAINING_DAY - DAYS_TO_VALIDATE
LAST_TRAINING_DAY - VAL_FIRST_DAY

val_fltr = df_first['InvoiceDate'] >= VAL_FIRST_DAY
trn, val = df_first[~val_fltr].reset_index(drop = True), df_first[val_fltr].reset_index(drop = True)

trn.shape
val.shape

TARGET_COL=['UnitPrice']
features=df_first.columns.to_list()
features.remove('UnitPrice')
features.remove('InvoiceDate')
features.remove('Description')

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

lgbm = LGBMRegressor(n_estimators=1000)
rf = RandomForestRegressor()
lr=LinearRegression(fit_intercept=True)

lgbm.fit(trn[features], trn['UnitPrice'], eval_set = [(val[features], val['UnitPrice'])], verbose = 50, early_stopping_rounds=200, eval_metric='rmse')

#rmse: 10.3029	valid_0's l2: 106.149 base line model split on day diff

from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score,hamming_loss
# from sklearn.preprocessing import LabelEncoder
# from sklearn.impute import SimpleImputer as Imputer
from sklearn.preprocessing import StandardScaler


x_train,x_test,y_train,y_test=train_test_split(df_first[features],df_first['UnitPrice'],test_size=0.30,random_state=32)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


lgbm.fit(x_train, y_train,eval_set = [(x_test, y_test)], verbose = 50, early_stopping_rounds=200, eval_metric='rmse')
#LGBM rmse for : rmse: 4.11437
lgbm_pred_train=lgbm.predict(x_train)
lgbm_pred=lgbm.predict(x_test)
lgbm_rmse_train=rmse(y_train,lgbm_pred_train)
lgbm_rmse=rmse(y_test,lgbm_pred)  ### 
print("LGBM rmse for train in {}  and test is {} :".format(lgbm_rmse_train,lgbm_rmse))  
#LGBM rmse for train in 3.1538854595543726  and test is 4.114373392062998 
fi=pd.Series(index=features,data=lgbm.feature_importances_)
fi.sort_values(ascending=False)[-20:][::-1].plot(kind='barh')
lgbm_test_pred=lgbm.predict(scaler.transform(test[features]))
#ss['label_lgbm']=lgbm_test_pred
ss.head()
test.head()
df_first[df_first['StockCode']==3216]
#Random forest base
rf.fit(x_train, y_train)
rf_pred_train=rf.predict(x_train)
rf_pred=rf.predict(x_test)
rf_rmse_train=rmse(y_train,rf_pred_train)
rf_rmse=rmse(y_test,rf_pred)  ### 
print("rf rmse for train is :" ,rf_rmse_train ," and rf test =",rf_rmse)  
#rf rmse for train is : 1.4963477856110567  and rf test = 2.8967685134559877
#that is model is overfitting


# fi=pd.Series(index=features,data=rf.feature_importances_)
# fi.sort_values(ascending=False)[-20:][::-1].plot(kind='barh')
# rf_test_pred=rf.predict(scaler.transform(test[features]))
# ss['UnitPrice']=rf_test_pred
# ss.head()
# ss.shape
# ss.to_csv("submission_v1_rf.csv")
#
#Random forest base

#linear regression base
lr.fit(x_train, y_train)
lr_pred=lr.predict(x_test)
lr_rmse=rmse(y_test,lr_pred)  ### 
print("lr rmse for test :" ,lr_rmse)#  lr rmse for : 7.171250042532457
lr_pred_train=lr.predict(x_train)
lr_rmse_train=rmse(y_train,lr_pred_train)  ### 
print("lr rmse for train :" ,lr_rmse_train)#  lr rmse for : 8.106859214167278
#lr rmse for train : 8.106859214167278
#lr rmse for test: 7.171250042532457
#linear regression base
#LGBM  tuned not performing well
model = LGBMRegressor(bagging_fraction=0.7, bagging_frequency=4, boosting_type='gbdt',
              class_weight=None, colsample_bytree=1.0, feature_fraction=0.5,
              importance_type='split', learning_rate=0.1, max_depth=30,
              min_child_samples=20, min_child_weight=30, min_data_in_leaf=70,
              min_split_gain=0.0001, n_estimators=20, n_jobs=-1,
              num_leaves=1400, objective=None, random_state=12, reg_alpha=0.0,
              reg_lambda=0.0, silent=True, subsample=1.0,
              subsample_for_bin=25000, subsample_freq=0)
#
model.fit(x_train, y_train,eval_set = [(x_test, y_test)], verbose = 50, early_stopping_rounds=200, eval_metric='rmse')

model_pred_train=model.predict(x_train)
model_rmse_train=rmse(y_train,model_pred_train) # 7.102986544548663
model_pred=model.predict(x_test)
model_rmse=rmse(y_test,model_pred)  ###  6.259807473591305
print("model rmse for :" ,model_rmse)  
fi=pd.Series(index=features,data=model.feature_importances_)
fi.sort_values(ascending=False)[-20:][::-1].plot(kind='barh')
#LGBM  tuned 

rf = RandomForestRegressor(n_estimators=26,min_samples_leaf=8,random_state=12)
rf.fit(x_train, y_train)
y_train_predicted = rf.predict(x_train)
y_test_predicted = rf.predict(x_test)
rmse_train = rmse(y_train, y_train_predicted)
rmse_test = rmse(y_test, y_test_predicted)
print("Esimator: {} Train rmse: {} Test rmse: {}".format(rf.n_estimators, rmse_train, rmse_test))
#Esimator: 26 Train rmse: 4.353284682388595 Test rmse: 3.479620865688163
rf_test_pred=rf.predict(scaler.transform(test[features]))
ss['UnitPrice']=rf_test_pred
ss.head()
ss.shape
ss.to_csv("submission_v3_rf.csv")
fi=pd.Series(index=features,data=rf.feature_importances_)
fi.sort_values(ascending=False)[-20:][::-1].plot(kind='barh')
#


features=df_first.columns.to_list()
features.remove('UnitPrice')
features.remove('InvoiceDate')
# features.remove('Description')
features.remove('InvoiceNo')
x_train,x_test,y_train,y_test=train_test_split(df_first[features],df_first['UnitPrice'],test_size=0.30,random_state=32)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

rf = RandomForestRegressor(n_estimators=26,min_samples_leaf=4,random_state=12)
rf.fit(x_train, y_train)
y_train_predicted = rf.predict(x_train)
y_test_predicted = rf.predict(x_test)
rmse_train = rmse(y_train, y_train_predicted)
rmse_test = rmse(y_test, y_test_predicted)
print("Esimator: {} Train rmse: {} Test rmse: {}".format(rf.n_estimators, rmse_train, rmse_test))
#Esimator: 26 Train rmse: 4.353284682388595 Test rmse: 3.479620865688163
rf_test_pred=rf.predict(scaler.transform(test[features]))
ss['UnitPrice']=rf_test_pred
ss.head()
ss.shape
ss.to_csv("submission_v3_rf.csv")
fi=pd.Series(index=features,data=rf.feature_importances_)
fi.sort_values(ascending=False)[-20:][::-1].plot(kind='barh')
#
features=['StockCode', 'Description', 'Quantity', 'CustomerID', 'Country']

x_train,x_test,y_train,y_test=train_test_split(df_first[features],df_first['UnitPrice'],test_size=0.30,random_state=32)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

rf = RandomForestRegressor(n_estimators=27,min_samples_leaf=5,random_state=12)
rf.fit(x_train, y_train)
y_train_predicted = rf.predict(x_train)
y_test_predicted = rf.predict(x_test)
rmse_train = rmse(y_train, y_train_predicted)
rmse_test = rmse(y_test, y_test_predicted)
print("Esimator: {} Train rmse: {} Test rmse: {}".format(rf.n_estimators, rmse_train, rmse_test))
#Esimator: 27 Train rmse: 4.414171769423885 Test rmse: 3.286181933929725
rf_test_pred=rf.predict(scaler.transform(test[features]))
ss['UnitPrice']=rf_test_pred
ss.head()
ss.shape
ss.to_csv("submission_v4_rf.csv")
fi=pd.Series(index=features,data=rf.feature_importances_)
fi.sort_values(ascending=False)[-20:][::-1].plot(kind='barh')
#





for iter in range(5):
    print(rf.min_samples_leaf)
    rf.fit(x_train, y_train)
    y_train_predicted = rf.predict(x_train)
    y_test_predicted = rf.predict(x_test)
    rmse_train = rmse(y_train, y_train_predicted)
    rmse_test = rmse(y_test, y_test_predicted)
    print("min_samples_leaf: {} Train rmse: {} Test rmse: {}".format(iter, rmse_train, rmse_test))
    rf.min_samples_leaf += 1
    rf.n_estimators+=1
features=df_first.columns.to_list()
features.remove('UnitPrice')
features.remove('InvoiceDate')
# features.remove('Description')
features.remove('InvoiceNo')
#Esimator: 26 Train rmse: 4.508825604469584 Test rmse: 3.4671350224607957
features=df_first.columns.to_list()
features.remove('UnitPrice')
features.remove('InvoiceDate')
# features.remove('Description')
features.remove('InvoiceNo')
#features.remove('CustomerID')
#Esimator: 26 Train rmse: 4.472746730821914 Test rmse: 3.4792972623290535
features=df_first.columns.to_list()
features.remove('UnitPrice')
features.remove('InvoiceDate')
# features.remove('Description')
features.remove('InvoiceNo')
#features.remove('CustomerID')
features.remove('Country')
Esimator: 26 Train rmse: 4.506438142442171 Test rmse: 3.4883103527516948

features=df_first.columns.to_list()
features.remove('UnitPrice')
features.remove('InvoiceDate')
# features.remove('Description')
features.remove('InvoiceNo')
#features.remove('CustomerID')
#features.remove('Country')
features.remove('year')
Esimator: 26 Train rmse: 4.510026661912849 Test rmse: 3.467258278813946





x_train,x_test,y_train,y_test=train_test_split(df_first[features],df_first['UnitPrice'],test_size=0.30,random_state=32)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

rf = RandomForestRegressor(n_estimators=26,min_samples_leaf=8,random_state=12)
rf.fit(x_train, y_train)
y_train_predicted = rf.predict(x_train)
y_test_predicted = rf.predict(x_test)
rmse_train = rmse(y_train, y_train_predicted)
rmse_test = rmse(y_test, y_test_predicted)
print("Esimator: {} Train rmse: {} Test rmse: {}".format(rf.n_estimators, rmse_train, rmse_test))  
