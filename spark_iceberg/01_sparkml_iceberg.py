import mlflow.spark
import os
import warnings
import sys
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import logging
import json
import shutil
import datetime
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
import cml.data_v1 as cmldata

#Edit your connection name here:
CONNECTION_NAME = "go01-aw-dl"

conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

training_df = spark.createDataFrame(
[
    ("0", "a b c d e spark", 1.0),
    ("1", "b d", 0.0),
    ("2", "spark f g h", 1.0),
    ("3", "hadoop mapreduce", 0.0),
],
["id", "text", "label"],
)

def exp1(df):

    mlflow.set_experiment("sparkml-experiment-new")

    ##EXPERIMENT 1

    df.writeTo("spark_catalog.default.training").using("iceberg").createOrReplace()
    spark.sql("SELECT * FROM spark_catalog.default.training").show()

    ### SHOW TABLE HISTORY AND SNAPSHOTS
    spark.read.format("iceberg").load("spark_catalog.default.training.history").show(20, False)
    spark.read.format("iceberg").load("spark_catalog.default.training.snapshots").show(20, False)

    snapshot_id = spark.read.format("iceberg").load("spark_catalog.default.training.snapshots").select("snapshot_id").tail(1)[0][0]
    committed_at = spark.read.format("iceberg").load("spark_catalog.default.training.snapshots").select("committed_at").tail(1)[0][0].strftime('%m/%d/%Y')
    parent_id = spark.read.format("iceberg").load("spark_catalog.default.training.snapshots").select("parent_id").tail(1)[0][0]

    tags = {
      "iceberg_snapshot_id": snapshot_id,
      "iceberg_snapshot_committed_at": committed_at,
      "iceberg_parent_id": parent_id,
      "row_count": training_df.count()
    }

    ### MLFLOW EXPERIMENT RUN
    with mlflow.start_run() as run:

        maxIter=8
        regParam=0.01

        tokenizer = Tokenizer(inputCol="text", outputCol="words")
        hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
        lr = LogisticRegression(maxIter=maxIter, regParam=regParam)
        pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
        model = pipeline.fit(training_df)

        mlflow.log_param("maxIter", maxIter)
        mlflow.log_param("regParam", regParam)

        #storage_path = "s3://go01-demo/pdefusco/tests/sparkmodels"
        mlflow.spark.log_model(model, artifact_path="pipe_artifacts", dfs_tmpdir="/home/cdsw/models-tests")

        #prediction = model.transform(test)
        mlflow.set_tags(tags)

    mlflow.end_run()

    experiment_id = mlflow.get_experiment_by_name("sparkml-experiment-new").experiment_id
    runs_df = mlflow.search_runs(experiment_id, run_view_type=1)

    return runs_df

exp1(training_df)

test_df = spark.createDataFrame(
[
    ("0", "a b c d e spark"),
    ("1", "b d"),
    ("2", "spark f g h"),
    ("3", "hadoop mapreduce"),
],
["id", "text"],
)

from pyspark.ml import PipelineModel

# Path where the pipeline model was saved
logged_model = '/home/cdsw/.experiments/yvyi-t7tu-8u5z-ezdm/omc7-qex7-jm6y-i5zv/artifacts/pipe_artifacts/sparkml/'

# Load the fitted pipeline model
loaded_pipeline_model = PipelineModel.load(logged_model)

# Now you can use the model to make predictions
predictions = loaded_pipeline_model.transform(test_df)
