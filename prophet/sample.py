import traceback
import json
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST
from mlflow.pyfunc import load_model
from mlflow.pyfunc import scoring_server
from io import StringIO
from mlflow.utils.proto_json_utils import NumpyEncoder, _get_jsonable_obj
import cml.models_v1 as models
import cml.metrics_v1 as metrics

logged_model = "/home/cdsw/.experiments/dei5-16qg-m1sa-58q7/huug-qmdg-weoe-waxo/artifacts/prophet_model"

def predictions_to_json(raw_predictions, output):
    predictions = _get_jsonable_obj(raw_predictions, pandas_orient="records")
    return predictions

@models.cml_model(metrics=True)
def predict_with_metrics(args):
    prediction = predict_actual(args)
    metrics.track_metric("prediction" ,prediction)
    return prediction

@models.cml_model
def predict(args):
    return predict_actual(args)

def predict_actual(targs):
    model = load_model(logged_model)
    args = json.dumps(targs)
    data = scoring_server.infer_and_parse_json_input(json_input=args, schema=None)
    try:
        raw_predictions = model.predict(data)
        return predictions_to_json(raw_predictions, None)
    except MlflowException as e:
        _handle_serving_error(
            error_message=e.message, error_code=BAD_REQUEST, include_traceback=False
        )


def _handle_serving_error(error_message, error_code, include_traceback=True):
    """
    Logs information about an exception thrown by model inference code that is currently being
    handled and reraises it with the specified error message. The exception stack trace
    is also included in the reraised error message.

    :param error_message: A message for the reraised exception.
    :param error_code: An appropriate error code for the reraised exception. This should be one of
                       the codes listed in the `mlflow.protos.databricks_pb2` proto.
    :param include_traceback: Whether to include the current traceback in the returned error.
    """
    if include_traceback:
        traceback_buf = StringIO()
        traceback.print_exc(file=traceback_buf)
        traceback_str = traceback_buf.getvalue()
        e = MlflowException(
            message=error_message, error_code=error_code, stack_trace=traceback_str
        )
    else:
        e = MlflowException(message=error_message, error_code=error_code)
    reraise(MlflowException, e)
    
    
payload = {"dataframe_split": {"columns": ["ds"], "data": [["1992-01-01T00:00:00"], ["1992-02-01T00:00:00"], ["1992-03-01T00:00:00"], ["1992-04-01T00:00:00"], ["1992-05-01T00:00:00"], ["1992-06-01T00:00:00"], ["1992-07-01T00:00:00"], ["1992-08-01T00:00:00"], ["1992-09-01T00:00:00"], ["1992-10-01T00:00:00"]]}}

predict_actual(json.dumps(payload))