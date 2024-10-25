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

import mlflow
import mlflow.pyfunc
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import *
from prophet import Prophet

# Define a UDF to apply Prophet forecasting
@pandas_udf(returnType=ArrayType(FloatType()), functionType=PandasUDFType.SCALAR)
def prophet_forecast_udf(history_pd: pd.DataFrame) -> pd.Series:
    model = Prophet()
    model.fit(history_pd)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast['yhat']

# Sample PySpark DataFrame
data = spark.createDataFrame([
    (1, "2023-01-01", 10),
    (1, "2023-01-02", 12),
    (1, "2023-01-03", 15),
    (2, "2023-01-01", 20),
    (2, "2023-01-02", 22),
    (2, "2023-01-03", 25)
], ["id", "ds", "y"])

#result = data.groupBy("id").applyInPandas(loaded_model.predict, schema="id int, yhat array<float>")

mlflow.set_experiment("prophet_pyspark_udf")
# Log the UDF as an MLflow model
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="prophet_udf",
        python_model=prophet_forecast_udf,
        pip_requirements=["prophet", "pyarrow"]
    )


import mlflow

logged_model = '/home/cdsw/.experiments/hok3-mid4-iq3h-j4gf/7idx-cypo-1cdo-jppt/artifacts/prophet_udf'

# Load model as a Spark UDF.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)

# Predict on a Spark DataFrame.
df.withColumn('predictions', loaded_model(*column_names)).collect()


# Load the model
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Apply the loaded model to the DataFrame
result = data.groupBy("id").applyInPandas(loaded_model.predict, schema="id int, yhat array<float>")

result.show()
