import os
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#importing the dataset

train = pd.read_csv('https://github.com/Yahiamostafa11/Bike-Sharing-Demand/raw/master/data/train.csv')
teast = pd.read_csv('https://github.com/Yahiamostafa11/Bike-Sharing-Demand/raw/master/data/test.csv')

# Boxplot of count

sns.boxplot(x='count', data-train,color=('mediumpurple')

plt.show()


# Histogram of count (It looks skew..)

sns.set_style('darkgrid')

sns.distplot(train['count'], bins=100,color=('green')

plt.show()


# Scatter plot between count & each numeric features

fields=[f for f in train]

fields=fields[5:-3]

print(fields)
    fig=plt.figure(figsize=(17,3))
        for i,f in enumerate(fields):

            ax=fig.add_subplot(1,4,i+1)

            ax.scatter(train[f], train['count'])

            ax.set_ylabel('count')

            ax.set_xlabel(f) I

plt.show()


# Boxplot between count & each categorical features

fig, axes=plt.subplots(nrows=2,ncols=2)

fig.set_size_inches(20,10)

sns.boxplot(data=train,y='count',x='season', ax=axes[0][0])

sns.boxplot(data=train,y='count',x='holiday', ax=axes[0][1])

sns.boxplot(data=train,y='count',x='workingday', ax=axes [1][0]) sns.boxplot(data=train,y='count',x='weather', ax=axes[1][1])

axes[0][0].set(xlabel='season', ylabel='count')

axes[0][1].set(xlabel='holiday',ylabel='count')

axes[1][0].set(xlabel='workingday',ylabel='count')

axes[1][1].set(xlabel='weather',ylabel='count')


# Correlation between each features

plt.figure(figsize=(10,10))

sns.heatmap(train.corr('pearson'), vmin=-1, vmax=1,cmap='coolwarm', annot=True, square=True)


# Convert datetime column to each elements (year, month, day, dayofweek, hour)

train['datetime']=pd.to_datetime(train['datetime'])

test['datetime']=pd.to_datetime(test['datetime'])

train.head()



def split_datetime(df):
    df['year']=df['datetime'].apply(lambda t:t.year)
    df['month']=df['datetime'].apply(lambda t:t.month)
    df['day']=df['datetime'].apply(lambda t:t.day)
    df['dayofweek'] df['datetime'].apply(lambda t:t.dayofweek)
    df['hour']=df['datetime'].apply(lambda t:t.hour)
    df=df.drop(['datetime'], axis=1)

return df

train=split_datetime(train)

test-split_datetime(test)

train-train.drop(['casual', 'registered'], axis-1

train.head()


#Boxplot between count & each categorical features

fig, axes-plt.subplots(nrows=1,ncols=3)

fig.set_size_inches (25,5)

sns.barplot(data-train,x='year',y='count', ax-axes[0])

sns.barplot(data=train,x='month',y='count',ax-axes[1])

sns.pointplot(data-train,x='hour',y='count', ax-axes[2], hue='dayofweek')


# Count column looks skew.

sns.distplot(train['count'])


# Take a log for count column

train['count']=np.log1p(train['count'])

sns.distplot(train['count'])

#Eliminate outliers (with residual less than stdev*3)

train-train[np.abs(train['count'])-train['count'].mean() <=(3*train['count'].std())]

# Boxplot of count

sns.boxplot(x='count',data=train,color=('mediumpurple')
plt.show()

# eliminate outlienrs
fig=plt.figure(figsize=(15,15))

for i,f1 in enumerate(fields):
    for j,2f in enumerate(fields):
        idx=i*len(fields)+i+1
        ax=fig.add_subplot(len(fields),len(fields),idx)
        ax.scatter(train[f1],train[f2])
        ax.set_ylabel(f1)
        ax.set_xlabel(f2)

plt.show()

drop_idx=train[(train['atemp']>20) & (train['atemp']<40) & (train['atemp']>10) & (train['atemp']<20)].index
train=train.drop(drop_idx)

# standard scaling

from sklearn.preprocessing import MinMaxScaler
def sclaing(df):
    scaler=MinMaxScaler()
    num_cols=['temp','atemp','humidity','windspeed']
    df[num_cols]=scaler.fit_transform(df[num_cols])
    return df

train=scaling(train) 
teat=scaling(test)

train.head()


# split tarin & test

from sklearn.model_selection import train_test_split

x_train,x-test,y_train,y_test=train_test_split(train.drop(['count'],axis=1),train['count'],test_size=0.3)

# define metric
def rmsle(y,pred):
    log_y=np.log1p(y)
    log_pred=np.log1p(pred)
    squared_error=(log_y-log_pred)**2
    rmsle=(np.sqrt(np.mean(squared_error)))
    return rmsle

# model selection

from sklearn.linear_model import linearRegraton,Ridge,lasso
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBRegressor

from sklearn.model_selection import GridSearchCV


# creat the function

def evaluat(reg_cls,params=None):
    reg=reg_cls()
    if params:
        reg=GridSearchCV(reg,param_grid=params,refit=True)
    reg.fit(X_train,y_train)
    pred=reg.predict(X_test)

    y_test_exp=np.expm(y_test)
    pred_exp=np.expm(pred)
    print('\n',reg_cls)
    if params:
        print(reg.best_params_)
        reg=reg.best_estimator_
    print(rmsle(y_teat_exp,pred_exp))
    return reg,pred_exp

# applay the function

lr_reg,pred_lr=evaluate(LinearRegression)
rg_reg,pred_rg=evaluate(Ridge)
ls_reg,pred_ls=evaluate(lasso)
rf_reg,pred_rf=evaluate(RandomForestRegressor)
gb_reg,pred_gb=evaluate(GradientBoostingRegressor)
xg_reg,pred_xg=evaluate(XGBRegressor)
lg_reg,pred_lg=evaluate(LGBMRegressor)

params={'n_estimators':[100*i for i in range(1,6)]}
xg_reg,pred_xg=evaluate(XGBRegressor,params)
lg_reg,pred_lg=evaluate(LGBMRegressor,params)

# revewing function

def feature_importance(reg):
    plt.figure(figsize=(20,10))
    print(type(reg))
    df=pd.DataFarme(sorted(zip(x_train.columns,reg.feature_importance_)),columns=['features','values'])
    sns.barplot(x='vlaues',y='features',data=df.sort_values(by='values',ascending=True))
    plt.show()

# applay the function

feature_importance(xg_reg)

#lightGBMRegressor 

feature_importance(lg_reg)

# submitt the resulte

submission=pd.read_csv("https://github.com/Yahiamostafa11/Bike-Sharing-Demand/raw/master/data/sampleSubmission.csv")
submission.head()

# pred = xg_reg.predict(test)

pred=lg_reg.predict(test)

pred_exp=np.expm1(pred)

print(pred_exp)

# submitt the resulte

submission.loc[:,'count']=pred_exp


