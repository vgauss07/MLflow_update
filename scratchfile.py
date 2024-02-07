# connect to MLflow tracking URI
# URI: Uniform Resource Identifier
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')

# to find the current tracking URL of mlflow
mlflow.tracking.get_tracking_uri()

# create a new experiment
exp_id = mlflow.create_experiment("Loan_Prediction")

# activate the experiment so that
# the run will be captured in the provided
# experiment
mlflow.set_experiment()


# for single iteration
run = mlflow.start_run()

# for multiple iterations
# with mlflow.start_run(run_name="") as run:
# mlflow.end_run()

n_estimators = 100
mlflow.log_param("n_estimators:", n_estimators)

accuracy = 0.8
mlflow.log_metric("accuracy", accuracy)


# set labels for identification
with mlflow.start_run():
    mlflow.set_tag("model_version", "0.1.0")