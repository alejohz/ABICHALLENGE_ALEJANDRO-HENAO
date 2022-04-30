#! /usr/bin/python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.manifold import TSNE

import time
import io

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import collections


# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


import warnings
warnings.filterwarnings("ignore")

# AWS libraries
import boto3
from mpu.aws import _s3_path_split



def s3_read(source, profile_name=None):
    """
    Read a file from an S3 source.

    Parameters:
    source : str
        Path starting with s3://, e.g. 's3://bucket-name/key/foo.bar'
    profile_name : str, optional
        AWS profile

    Returns:
    content : bytes
    """
    session = boto3.Session(profile_name=profile_name)
    s3 = session.client('s3')
    bucket_name, key = _s3_path_split(source)
    s3_object = s3.get_object(Bucket=bucket_name, Key=key)
    body = s3_object['Body']
    return pd.read_csv(io.BytesIO(body.read()))

data = s3_read('s3://abi-datalake/raw/creditcard.csv')



class Model:
    def __init__(self, datafile = None):
        '''
        Initializes the Model class
        Parameters:

        Returns:
        '''
        self.df = pd.read_csv(datafile)
    

    def __repr__(self):
        return 'ML Model Class'
    
    def show(self, rows=5):
        return self.df.head(rows)

    def description(self):
        return self.df.describe()

    def check_nan_values(self):
        nan_values = self.df.isnull().sum().max()
        
        if nan_values == 0:
            return 'NO NAN VALUES'
        else:
            nan_columns = self.df.columns[self.df.isna().any()].tolist()
            return f'This list of columns has NAN VALUES: {nan_columns}'

    
    def scaling(self):
        scaler = StandardScaler()
        self.df_scaled = self.df.copy(deep=True)
        self.df_scaled['scaled_amount'] = scaler.fit_transform(self.df_scaled['Amount'].values.reshape(-1,1))
        self.df_scaled['scaled_time'] = scaler.fit_transform(self.df_scaled['Time'].values.reshape(-1,1))
        self.df_scaled.drop(['Time','Amount'], axis=1, inplace=True)
        scaled_amount = self.df_scaled['scaled_amount']
        scaled_time = self.df_scaled['scaled_time']
        self.df_scaled.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
        self.df_scaled.insert(0, 'scaled_amount', scaled_amount)
        self.df_scaled.insert(1, 'scaled_time', scaled_time)


    def split(self, test_size):
        self.X = self.df_scaled.drop('Class', axis=1)
        self.y = self.df_scaled['Class']

        sss = StratifiedShuffleSplit(n_splits=5, random_state=42, shuffle=False, test_size=test_size)
        for train_index, test_index in sss.split(X, y):
            self.X_train, self.X_test = self.X.iloc[train_index].values, self.X.iloc[test_index].values
            self.y_train, self.y_test = self.y.iloc[train_index].values, self.y.iloc[test_index].values

    def random_under_sampling(self):
        self.df_rus = self.df_scaled.sample(frac=1).copy(deep=True)

        self.fraud_df = self.df_rus.loc[self.df_rus['Class'] == 1]
        self.non_fraud_df = self.df_rus.loc[self.df_rus['Class'] == 0][:len(self.fraud_df)]

        self.nd_df = pd.concat([self.fraud_df, self.non_fraud_df])

        self.new_df = self.nd_df.sample(frac=1, random_state=42).copy(deep=True)
