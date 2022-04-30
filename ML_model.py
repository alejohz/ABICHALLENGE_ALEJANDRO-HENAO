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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report


import warnings
warnings.filterwarnings("ignore")

# AWS libraries
import boto3
from mpu.aws import _s3_path_split



class Model:
    def __init__(self, datafile = None):
        '''
        Initializes the Model class
        Parameters:

        Returns:
        '''
        self.df = datafile
        self.df_scaled = self.df.copy(deep=True)
    

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
        return self.df_scaled
        

    def random_under_sampling(self):
        self.df_rus = self.df_scaled.sample(frac=1).copy(deep=True)

        self.fraud_df = self.df_rus.loc[self.df_rus['Class'] == 1]
        self.non_fraud_df = self.df_rus.loc[self.df_rus['Class'] == 0][:len(self.fraud_df)]

        self.nd_df = pd.concat([self.fraud_df, self.non_fraud_df])

        self.new_df = self.nd_df.sample(frac=1, random_state=42).copy(deep=True)
        return self.new_df
    
    def split(self, test_size):
        self.X = self.new_df.drop('Class', axis=1)
        self.y = self.new_df['Class']

        sss = StratifiedShuffleSplit(n_splits=5, random_state=42, test_size=test_size)
        for train_index, test_index in sss.split(self.X, self.y):
            self.X_train, self.X_test = self.X.iloc[train_index].values, self.X.iloc[test_index].values
            self.y_train, self.y_test = self.y.iloc[train_index].values, self.y.iloc[test_index].values
        return self.X_train, self.X_test, self.y_train, self.y_test

    def compile_fit(self):
        self.X = self.new_df.drop('Class', axis=1)
        self.y = self.new_df['Class']
        self.model = xgb.XGBClassifier()
        self.model.fit(self.X_train, self.y_train)
        self.training_score = cross_val_score(self.model, self.X_train, self.y_train, cv=5)
        return self.training_score.mean()
    
    def predict(self, input_value=None):
        '''
        predict instance of class Model

        Parameters:
        input_value: list or 1d array
            it must have shape (1,30)
        '''
        if input_value == None:
            self.result = self.model.predict(self.X_test)
        else: 
            self.result = self.model.predict(np.array([input_value]))
        return self.result
    
    def validate(self):
        print(classification_report(self.y_test, self.result))


if __name__ == '__main__':
    model_instance = Model(datafile=data)
    df_scaled = model_instance.scaling()
    model_instance.split(0.2)
    model_instance.random_under_sampling()
    sc = model_instance.compile_fit()
    print(sc)
    ts = model_instance.predict()
    print(ts)
