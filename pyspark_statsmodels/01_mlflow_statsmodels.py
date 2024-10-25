import pandas as pd
import mlflow
import mlflow.statsmodels
from statsmodels.tsa.arima.model import ARIMA
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql import SparkSession
import pyspark.pandas as ps
from pyspark.sql.types import FloatType, StructType, StructField, DoubleType, StringType

# Initialize Spark session
#spark = SparkSession.builder.appName("ARIMA Example").getOrCreate()
import cml.data_v1 as cmldata

# Sample in-code customization of spark configurations
from pyspark import SparkContext
SparkContext.setSystemProperty('spark.executor.cores', '4')
SparkContext.setSystemProperty('spark.executor.memory', '8g')

USERNAME = os.environ["PROJECT_OWNER"]
DBNAME = "STATS_MODELS_MLFLOW_"+USERNAME
CONNECTION_NAME = "paul-aug26-aw-dl"

conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

# Sample usage to run query through spark
EXAMPLE_SQL_QUERY = "show databases"
spark.sql(EXAMPLE_SQL_QUERY).show()

import statsmodels.tsa.api as sm
import numpy as np
import pandas as pd

data = spark.sql("SELECT * FROM SPARK_CATALOG.{0}.ARIMA_TS_{1}".format(DBNAME, USERNAME))

data.printSchema()

data = data.withColumn("date", F.to_date("date"))


# read the entire data as spark dataframe
#data = spark.read.format('csv')\
#  .options(header='true', inferSchema='true')\
#  .load('/home/cdsw/train.csv')\
#  .select('Store','Dept','Date','Weekly_Sales')

## basic data cleaning before implementing the pandas udf
##removing Store - Dept combination with less than 2 years (52 weeks ) of data

selected_com = data.groupBy("code")\
                      .count()\
                      .filter("count > 104")\
                      .select("code")

data_selected_store_departments = data.join(selected_com,['code'],'inner')

##pandas udf
schema = StructType([StructField('code', StringType(), True),
                     StructField('credit_card_balance_forecast', DoubleType(), True)])

@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def holt_winters_time_series_udf(data):

    data.set_index('date',inplace = True)
    time_series_data = data['credit_card_balance']

    ##the model
    model_monthly = sm.ExponentialSmoothing(np.asarray(time_series_data),trend='add').fit()

    ##forecast values
    forecast_values = pd.Series(model_monthly.forecast(2),name = 'fitted_values')

    return pd.DataFrame({'code': [str(data.code.iloc[0])],\
                          'credit_card_balance_forecast': [forecast_values[0]]})

mlflow.set_experiment("pyspark arima new")
# Log the UDF as an MLflow model
with mlflow.start_run():
  ##aggregating the forecasted results in the form of a spark dataframe
  forecasted_spark_df = data_selected_store_departments.groupby('code').apply(holt_winters_time_series_udf)

  ## to see the forecasted results
  forecasted_spark_df.show(10)

  mlflow.pyfunc.log_model(
      artifact_path="model",
      python_model=holt_winters_time_series_udf,
      input_example=pd.DataFrame({"code": [str("A")], "credit_card_balance_forecast":[float(24924.5)]}),  # Provide an example input
      conda_env={"channels": ["defaults"], "dependencies": ["pandas"]}
  )

  #print(model_monthly.summary())

  mlflow.end_run()
