import pandas as pd

import logging
import boto3
from botocore.exceptions import ClientError


from mpu.aws import _s3_path_split
import io
from sklearn import datasets
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import train_test_split

import xgboost as xgb

class Read_Upload:

    def __init__(self, key=None):
        """
        Initializing class object for reading and uploading other scripts, helper class
        """
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
        self.body = s3_object['Body']
        return self.body.read()
    

    def s3_upload(self, bucket_name, file, object_name):
        """Upload a file to an S3 bucket

        Parameters:
        file_name: File to upload
        bucket: Bucket to upload to
        object_name: S3 object name
        :return: True if file was uploaded, else False
        """
        try:
            response = self.s3.upload_file(file, bucket_name, object_name)
        except ClientError as e:
            logging.error(e)
            return False
        return True

if __name__ == '__main__':
    rsu = Read_Upload()
    bucket = 'abi-datalake'

    #data = rsu.s3_read('s3://abi-datalake/raw/creditcard.csv')
    iris = datasets.load_iris() # Loading dataset
    X = iris.data # Spliting into dependent 
    y = iris.target # and independent variables

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # test_train splitting

    # use DMatrix for xgboost
    dtrain = xgb.DMatrix(X_train, label=y_train) # converting into dmatrix
    dtest = xgb.DMatrix(X_test, label=y_test)

    # use svmlight file for xgboost
    dump_svmlight_file(X_train, y_train, 'dtrain.svm', zero_based=True) #dumping svm file to upload
    dump_svmlight_file(X_test, y_test, 'dtest.svm', zero_based=True)
    
    train_path = "{}/{}".format("train",'dtrain.svm') #key and obbject name to upload to s3
    test_path = "{}/{}".format("test",'dtest.svm')

    rsu.s3_upload(bucket_name=bucket, file= 'dtrain.svm',object_name = train_path) # uploading to datalake 
    rsu.s3_upload(bucket_name=bucket, file=  'dtest.svm',object_name = test_path)
