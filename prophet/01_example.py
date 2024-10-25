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

import numpy as np
import pandas as pd
from prophet import Prophet, serialize
from prophet.diagnostics import cross_validation, performance_metrics
import mlflow

SOURCE_DATA = (
    "https://raw.githubusercontent.com/facebook/prophet/master/examples/example_retail_sales.csv"
)
np.random.seed(12345)

def extract_params(pr_model):
    params = {attr: getattr(pr_model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}
    return {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}

sales_data = pd.read_csv(SOURCE_DATA)

mlflow.set_experiment("prophet-forecast")
with mlflow.start_run():
    model = Prophet().fit(sales_data)
    params = extract_params(model)

    metrics_raw = cross_validation(
        model=model,
        horizon="365 days",
        period="180 days",
        initial="710 days",
        parallel="threads",
        disable_tqdm=True,
    )

    cv_metrics = performance_metrics(metrics_raw)
    metrics = cv_metrics.drop(columns=["horizon"]).mean().to_dict()

    # The training data can be retrieved from the fit model for convenience
    train = model.history

    model_info = mlflow.prophet.log_model(
        model, artifact_path="prophet_model", input_example=train[["ds"]].head(10)
    )
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

loaded_model = mlflow.prophet.load_model(model_info.model_uri)

forecast = loaded_model.predict(loaded_model.make_future_dataframe(60))
forecast = forecast[["ds", "yhat"]].tail(90)
print(f"forecast:\n${forecast.head(30)}")


#{"dataframe_split": {"columns": ["FL_VIVO_TOTAL", "TAXPIS", "TAXCOFINS", "TAXISS"], "data":[[35.5, 200.5, 30.5, 14.5]]}}

#{"dataframe_split": {"columns": ["ds"], "data": [["1992-01-01T00:00:00"], ["1992-02-01T00:00:00"], ["1992-03-01T00:00:00"], ["1992-04-01T00:00:00"], ["1992-05-01T00:00:00"], ["1992-06-01T00:00:00"], ["1992-07-01T00:00:00"], ["1992-08-01T00:00:00"], ["1992-09-01T00:00:00"], ["1992-10-01T00:00:00"]]}}