# Launch training job
# We use the Estimator from the SageMaker Python SDK
from sagemaker.session import Session
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost.estimator import XGBoost
import sagemaker

from read_upload import Read_Upload
class sagemaker_xgboost:
    def __init__(self):
        """
        Initializing class with needed variables

        bucket, hyperparams,instance type, output path, content type
        """
        self.bucket = 'abi-datalake'

        self.hyperparams = {
            "max_depth": "3",
            "eta": "0.3",
            "gamma": "4",
            "min_child_weight": "6",
            "subsample": "0.7",
            "objective": "multi:softprob",
            "num_class": "3",
            "num_round": "20",
            "verbosity": "2",
        }

        self.instance_type = "ml.m4.xlarge"
        self.output_path = "s3://{}/{}/".format(self.bucket, "output")
        self.content_type = "libsvm"

        self.role = 'iam_execution_role'

        self.session = Session()
        self.script_path = "script.py"
    def model_fit(self):
        """
        fit instance for XGBoost for classification of iris dataset
        """
        self.xgb_script_mode_estimator = XGBoost(
            entry_point=self.script_path,
            framework_version="1.5-1",  # Note: framework_version is mandatory
            hyperparameters=self.hyperparams,
            role=self.role,
            instance_count=2,
            instance_type=self.instance_type,
            output_path=self.output_path,
        )

        train_input = TrainingInput(
            "s3://{}/{}/".format(self.bucket, "train"), content_type=self.content_type
        )
        validation_input = TrainingInput(
            "s3://{}/{}/".format(self.bucket, "test"), content_type=self.content_type
        )

        self.xgb_script_mode_estimator.fit({"train": train_input, "validation": validation_input})
        self.predictor = self.xgb_script_mode_estimator.deploy(
            initial_instance_count=1, instance_type="ml.m4.xlarge"
        )

    def model_deploy(self):
        source = 's3://abi-datalake/sagemaker-xgboost-2022-05-01-20-18-16-799/source/sourcedir.tar.gz'
        model_data = 's3://abi-datalake/output/sagemaker-xgboost-2022-05-01-20-18-16-799/output/model.tar.gz'
        model = sagemaker.model.Model(
            image_uri='683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.5-1',
            model_data=model_data,
            role=self.role,
            source_dir = source,
            predictor_cls=sagemaker.xgboost.XGBoostPredictor)
        self.predictor = model.deploy(
            initial_instance_count=1,
            instance_type='ml.m4.xlarge'
            )
        return self.predictor

    def model_predict(self, input_data, predictor, testing=False):

        if testing:
            self.payload_path = "s3://{}/{}/{}".format(self.bucket, "test", "dtest.svm")
            rsu = Read_Upload()
            self.payload = rsu.s3_read(self.payload_path)
            runtime_client = self.session.sagemaker_runtime_client
            self.response = runtime_client.invoke_endpoint(
            EndpointName=predictor.endpoint_name, ContentType="text/libsvm", Body=self.payload
            )
            result = self.response["Body"].read()
        else:
            input_data = ",".join([str(a) for a in input_data])
            self.payload = input_data
            runtime_client = self.session.sagemaker_runtime_client
            self.response = runtime_client.invoke_endpoint(
            EndpointName=predictor.endpoint_name, ContentType="text/csv", Body=self.payload
            )
            result = self.response["Body"].read()
        
        return result
       
    def model_cancel(self, predictor):
        predictor.delete_model()
        predictor.delete_endpoint()

if __name__ == '__main__':
    pass

  