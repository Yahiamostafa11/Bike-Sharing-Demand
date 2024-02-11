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
