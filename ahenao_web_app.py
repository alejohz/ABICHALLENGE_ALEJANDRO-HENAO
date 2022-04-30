import pandas as pd
import numpy as np
import streamlit as st



# Launch training job
# We use the Estimator from the SageMaker Python SDK
from sagemaker.sklearn.estimator import XGBoost
from sagemaker import get_execution_role

FRAMEWORK_VERSION = "0.23-1"

xgboost_estimator = XGBoost(
    entry_point="script.py",
    role='iam_execution_role',
    instance_count=1,
    instance_type="ml.c5.xlarge",
    framework_version=FRAMEWORK_VERSION,
    base_job_name="xgb-scikit",

)

xgboost_estimator.fit(wait=False)