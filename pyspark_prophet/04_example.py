#****************************************************************************
# (C) Cloudera, Inc. 2020-2025
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

import os
import numpy as np
import pandas as pd
from datetime import datetime
from pyspark.sql.types import LongType, IntegerType, StringType
from pyspark.sql import SparkSession
import dbldatagen as dg
import dbldatagen.distributions as dist
from dbldatagen import FakerTextFactory, DataGenerator, fakerText
from faker.providers import bank, credit_card, currency
import mlflow
import pandas as pd
from prophet import Prophet
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import *
from pyspark.sql import functions as F

class DataGen:

  '''Class to Generate Data'''

  def __init__(self, username):
      self.username = username

  def dataGen(self, spark, shuffle_partitions_requested = 5, partitions_requested = 2, data_rows = 10000):
      """
      Method to create credit card transactions in Spark Df
      """

      # setup use of Faker
      FakerTextUS = FakerTextFactory(locale=['en_US'], providers=[bank])

      # partition parameters etc.
      spark.conf.set("spark.sql.shuffle.partitions", shuffle_partitions_requested)

      fakerDataspec = (DataGenerator(spark, rows=data_rows, partitions=partitions_requested)
                  .withColumn("ds", "timestamp", begin="2020-01-01 01:00:00",end="2024-12-31 23:59:00",interval="1 second", random=True)
                  .withColumn("y", "float", minValue=10, maxValue=100, random=True)
                  )
      df = fakerDataspec.build()

      return df

USERNAME = os.environ["PROJECT_OWNER"]

# Instantiate BankDataGen class
dg = DataGen(USERNAME)

# Create Banking Transactions DF
df = dg.dataGen(spark)
df = df.withColumn("ds", F.to_date("ds"))

schema = StructType([
   StructField("ds", DateType(), True),
   StructField("y", DoubleType(), True)
])

# Create UDF
@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def forecast_udf(df):
  model = Prophet()
  model.fit(df)
  future = model.make_future_dataframe(periods=30)
  forecast = model.predict(future)
  return forecast[['ds', 'y']]

# Apply UDF
result_df = df.groupBy().apply(forecast_udf)

# Log with MLflow
import mlflow.prophet

mlflow.set_experiment("prophet-model-on-pyspark")
with mlflow.start_run():
  # Log model parameters
  mlflow.log_params(model.get_params())

  # Log the model
  mlflow.prophet.log_model(model, "prophet_model")
