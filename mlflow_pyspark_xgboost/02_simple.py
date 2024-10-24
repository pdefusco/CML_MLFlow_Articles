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

from pyspark.sql import SparkSession
from xgboost.spark import SparkXGBClassifier
from pyspark.ml.linalg import Vectors

spark = SparkSession\
    .builder\
    .appName("SparkXGBoostClassifier Example")\
    .config("spark.hadoop.fs.s3a.s3guard.ddb.region", "us-east-2")\
    .config("spark.kerberos.access.hadoopFileSystems","s3a://go01-demo")\
    .config("spark.dynamic allocation.enabled", "false")\
    .config("spark.executor.cores", "4")\
    .config("spark.executor.memory", "4g")\
    .config("spark.executor.instances", "2")\
    .config("spark.driver.core","4")\
    .config("spark.driver.memory","4g")\
    .getOrCreate()

import os
print("https://spark-"+os.environ["CDSW_ENGINE_ID"]+"."+os.environ["CDSW_DOMAIN"])

df_train = spark.createDataFrame([
    (Vectors.dense(1.0, 2.0, 3.0), 0, False, 1.0),
    (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 1, False, 2.0),
    (Vectors.dense(4.0, 5.0, 6.0), 0, True, 1.0),
    (Vectors.sparse(3, {1: 6.0, 2: 7.5}), 1, True, 2.0),
], ["features", "label", "isVal", "weight"])

df_test = spark.createDataFrame([
    (Vectors.dense(1.0, 2.0, 3.0), ),
], ["features"])

xgb_classifier = SparkXGBClassifier(max_depth=5, missing=0.0,
    validation_indicator_col='isVal', weight_col='weight',
    early_stopping_rounds=1, eval_metric='logloss', num_workers=2)

xgb_clf_model = xgb_classifier.fit(df_train)

xgb_clf_model.transform(df_test).show()
