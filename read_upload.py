import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import logging
import boto3
from botocore.exceptions import ClientError

from sklearn.model_selection import StratifiedShuffleSplit

import os
from mpu.aws import _s3_path_split
import io


class Read_Upload:

    def __init__(self, key=None):
        self.sm_boto3 = boto3.client("sagemaker")
        self.session = boto3.Session()
        self.s3 = self.session.client('s3')
        if key:
            self.key = '/' + key

    def __repr__(self):
        return 'Class for reading and uploading to S3'

    def s3_read(self, source, profile_name=None):
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
        self.bucket_name, self.key = _s3_path_split(source) 
        s3_object = self.s3.get_object(Bucket=self.bucket_name, Key=self.key)
        body = s3_object['Body']
        self.file = pd.read_csv(io.BytesIO(body.read()))
        return self.file
    

    def s3_upload(self, file, object_name):
        """Upload a file to an S3 bucket

        Parameters:
        file_name: File to upload
        bucket: Bucket to upload to
        object_name: S3 object name
        :return: True if file was uploaded, else False
        """
        try:
            response = self.s3.upload_file(file, self.bucket_name, object_name)
        except ClientError as e:
            logging.error(e)
            return False
        return True

if __name__ == '__main__':
    rsu = Read_Upload()
    data = rsu.s3_read('s3://abi-datalake/raw/creditcard.csv')
    X_train, X_test, y_train, y_test = rsu.split(test_size=0.2)
    train = pd.concat(X_train, y_train)
    test = pd.concat(X_test, y_test)

    train_file_name = 'train.csv'
    test_file_name = 'test.csv'

    train_key = 'train/'
    test_key = 'test/'

    train.to_csv(train_file_name)
    test.to_csv(test_file_name)
    

    train_path_abs = os.path.dirname(os.path.join(os.path.abspath(train_file_name), train_file_name))
    test_path_abs = os.path.dirname(os.path.join(os.path.abspath(test_file_name), test_file_name))
    
    rsu.s3_upload(train_path_abs, train_key + train_file_name)
    rsu.s3_upload(test_path_abs, test_key + test_file_name)
