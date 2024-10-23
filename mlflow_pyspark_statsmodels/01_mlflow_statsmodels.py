import pandas as pd
import mlflow
import mlflow.statsmodels
from statsmodels.tsa.arima.model import ARIMA
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql import SparkSession
import pyspark.pandas as ps
from pyspark.sql.types import FloatType, StructType, StructField, DoubleType

# Initialize Spark session
#spark = SparkSession.builder.appName("ARIMA Example").getOrCreate()
import cml.data_v1 as cmldata

# Sample in-code customization of spark configurations
from pyspark import SparkContext
SparkContext.setSystemProperty('spark.executor.cores', '1')
SparkContext.setSystemProperty('spark.executor.memory', '2g')

CONNECTION_NAME = "go01-aw-dl"
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

# Sample usage to run query through spark
EXAMPLE_SQL_QUERY = "show databases"
spark.sql(EXAMPLE_SQL_QUERY).show()

import statsmodels.tsa.api as sm
import numpy as np
import pandas as pd

# read the entire data as spark dataframe
data = spark.read.format('csv')\
  .options(header='true', inferSchema='true')\
  .load('/home/cdsw/train.csv')\
  .select('Store','Dept','Date','Weekly_Sales')

## basic data cleaning before implementing the pandas udf
##removing Store - Dept combination with less than 2 years (52 weeks ) of data

selected_com = data.groupBy(['Store','Dept'])\
                      .count()\
                      .filter("count > 104")\
                      .select("Store","Dept")

data_selected_store_departments = data.join(selected_com,['Store','Dept'],'inner')

##pandas udf
schema = StructType([StructField('Store', StringType(), True),
                     StructField('Dept', StringType(), True),
                     StructField('weekly_forecast_1', DoubleType(), True),
                     StructField('weekly_forecast_2', DoubleType(), True)])

@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def holt_winters_time_series_udf(data):

    data.set_index('Date',inplace = True)
    time_series_data = data['Weekly_Sales']

    ##the model
    model_monthly = sm.ExponentialSmoothing(np.asarray(time_series_data),trend='add').fit()

    ##forecast values
    forecast_values = pd.Series(model_monthly.forecast(2),name = 'fitted_values')

    return pd.DataFrame({'Store': [str(data.Store.iloc[0])],\
                          'Dept': [str(data.Dept.iloc[0])],\
                          'weekly_forecast_1': [forecast_values[0]],\
                          'weekly_forecast_2':[forecast_values[1]]})

# Define your UDF
@pandas_udf("double", PandasUDFType.SCALAR)
def predict_udf(data: pd.Series) -> pd.Series:
    # Load your model here (e.g., using mlflow.pyfunc.load_model)
    model = ...
    return model.predict(data)

# Log the UDF as an MLflow model
with mlflow.start_run():
  ##aggregating the forecasted results in the form of a spark dataframe
  forecasted_spark_df = data_selected_store_departments.groupby(['Store','Dept']).apply(holt_winters_time_series_udf)

  ## to see the forecasted results
  forecasted_spark_df.show(10)

  mlflow.pyfunc.log_model(
      artifact_path="model",
      python_model=holt_winters_time_series_udf,
      input_example=pd.DataFrame({"Store": [str(1)], "Dept": [str(1)], "weekly_forecast_2":[float(24924.5)], "weekly_forecast_2":[float(24924.5)]}),  # Provide an example input
      conda_env={"channels": ["defaults"], "dependencies": ["pandas"]}
  )

  mlflow.end_run()
