#****************************************************************************
# (C) Cloudera, Inc. 2020-2023
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# #  Author(s): Paul de Fusco
#***************************************************************************/

import mlflow.pyfunc
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import *

def my_udf(col1: pd.Series, col2: pd.Series) -> pd.Series:
  return col1+col2


def my_udf(df: pd.DataFrame) -> pd.DataFrame:
    # Your UDF logic here
    return df

@pandas_udf(returnType=DoubleType(), functionType=PandasUDFType.SCALAR)
def wrapped_udf(*args):
    return my_udf(pd.DataFrame(args)).iloc[:, 0]



# Pandas UDF to add two columns
@pandas_udf("long", PandasUDFType.SCALAR)
def wrapped_udf:
    return col1 + col2

# Sample DataFrame
df = spark.createDataFrame([(1, 2), (3, 4)], ["col1", "col2"])
# Use the UDF in PySpark
df = df.withColumn("sum", add_cols("col1", "col2"))


class MyModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        return add_cols(model_input)

model = MyModel()

mlflow.set_experiment("model-from-udf")
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=model
)

import mlflow
logged_model = '/home/cdsw/.experiments/y1pt-21l1-fij6-jzrw/tsjx-t234-hnb9-mjga/artifacts/model'

# Load model as a Spark UDF.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)

#loaded_model = mlflow.pyfunc.load_model("model")
cols = ["col1", "col2"]
df.withColumn('predictions', loaded_model(*cols)).collect()
