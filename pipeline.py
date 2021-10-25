BASE_IMAGE = "gcr.io/google-appengine/python"
PROJECT_ID = 'lab1-328711'
Dataset = 'gs://perry-dataset/train.csv'

import kfp
import kfp.dsl as dsl
from kfp.v2 import compiler
import kfp.v2.dsl as dsl
from kfp.v2.dsl import (Artifact, Dataset,
Input, InputPath, Model, Output,OutputPath, Metrics,ClassificationMetrics)
import pandas as pd
from sklearn.linear_model import LinearRegression
import google.cloud.aiplatform as aip


@dsl.component (
base_image = BASE_IMAGE,
packages_to_install=["google-cloud-aiplatform","pandas"],)
def ingest_data(dataset_train: Output[Dataset]):
    import pandas as pd
    df = pd.read_csv(Dataset)
    dataset_train = df.to_csv(dataset_train.path)


@dsl.component (
base_image = BASE_IMAGE,
packages_to_install=[ "google-cloud-aiplatform", "scikit-learn","pandas"],
)
def train(dataset_train: Input[Dataset]):
    from sklearn.linear_model import LinearRegression
    import pandas as pd
    data = pd.read_csv(dataset_train.path)
    data = pd.get_dummies (data)
    data.fillna(value = 0 , inplace=True)
    Y= data['SalePrice']
    x= data.drop(['SalePrice'],axis=1)
    model = LinearRegression()
    model.fit(x,Y)


@dsl.pipeline(pipeline_root='gs://perry-dataset/pipeline' , name='ingest-data' )
def pipeline ():
    dataset_job = ingest_data()
    train_job = train(dataset_job.outputs["dataset_train"])


template_path="train.json"
compiler.Compiler().compile(
pipeline_func=pipeline, package_path=template_path
)


from kfp.v2.google.client import AIPlatformClient
api_client = AIPlatformClient(project_id=PROJECT_ID, region='europe-west4' )
response = api_client.create_run_from_job_spec("train.json".replace(" ", "_"),
pipeline_root='gs://perry-dataset/pipeline',
parameter_values={},

service_account="643685219419-compute@developer.gserviceaccount.com",

enable_caching=False
)