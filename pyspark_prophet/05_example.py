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


from pyspark.sql.types import StructType,StructField,StringType,TimestampType,ArrayType,DoubleType
from pyspark.sql.functions import current_date
from pyspark.sql.functions import pandas_udf, PandasUDFType
from fbprophet import Prophet
from datetime import datetime
import pandas as pd


result_schema = StructType([

    StructField('segment', StringType(), True),
    StructField('ds', TimestampType(), True),
    StructField('trend', DoubleType(), True),
    StructField('trend_upper', DoubleType(), True),
    StructField('trend_lower', DoubleType(), True),
    StructField('yearly', DoubleType(), True),
    StructField('yearly_upper', DoubleType(), True),
    StructField('yearly_lower', DoubleType(), True),
    StructField('yhat', DoubleType(), True),
    StructField('yhat_upper', DoubleType(), True),
    StructField('yhat_lower', DoubleType(), True),
    StructField('multiplicative_terms', DoubleType(), True),
    StructField('multiplicative_terms_upper', DoubleType(), True),
    StructField('multiplicative_terms_lower', DoubleType(), True),
    StructField('additive_terms', DoubleType(), True),
    StructField('additive_terms_upper', DoubleType(), True),
    StructField('additive_terms_lower', DoubleType(), True),

    ])


@pandas_udf(result_schema, PandasUDFType.GROUPED_MAP)
def forecast_loans(df):

    def prophet_model(df,test_start_date):

        df['ds'] = pd.to_datetime(df['ds'])

        # train
        ts_train = (df
                    .query('ds < @test_start_date')
                    .sort_values('ds')
                    )
        # test
        ts_test = (df
                   .query('ds >= @test_start_date')
                   .sort_values('ds')
                   .drop('y', axis=1)
                   )

        print(ts_test.columns)

        # instantiate the model, configure the parameters
        model = Prophet(
            interval_width=0.95,
            growth='linear',
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=True,
            seasonality_mode='multiplicative'
        )

        # fit the model

        model.fit(ts_train.loc[:,['ds','y']])

        # configure predictions
        future_pd = model.make_future_dataframe(
            periods=len(ts_test),
            freq='W')

        # make predictions
        results_pd = model.predict(future_pd)
        results_pd = pd.concat([results_pd,df['segment']],axis = 1)

        return pd.DataFrame(results_pd, columns = result_schema.fieldNames())

    # return predictions
    return prophet_model(df, test_start_date= '2019-03-31')




results =df3.groupBy('segment').apply(forecast_loans)
