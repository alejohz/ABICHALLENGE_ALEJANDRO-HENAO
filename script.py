import argparse
import joblib
import os

import numpy as np
import pandas as pd

from ML_model import Model

from read_upload import Read_Upload
rsu = Read_Upload()

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


if __name__ == "__main__":

    data = rsu.s3_read('s3://abi-datalake/raw/creditcard.csv')
    m = Model(datafile=data)
    df_scaled = m.scaling()
    m.random_under_sampling()
    print("building training and testing datasets")
    X_train, X_test, y_train, y_test = m.split(test_size=0.2)

    
    # train
    print("training model")

    m.compile_fit()

    # print abs error
    print("validating model")
    result = m.predict()

    m.validate()

    # persist model
    path = "./model.joblib"
    joblib.dump(m, path)
    print("model persisted at " + path)