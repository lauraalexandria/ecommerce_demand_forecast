import datetime
import logging
import os

import click
import pandas as pd
import psycopg
from dotenv import load_dotenv
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.report import Report
from prefect import task  # flow,

import mlflow

load_dotenv(dotenv_path=".env")

logging.basicConfig(
    level=logging.INFO,
    filename="app.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# pylint: disable=c-extension-no-member
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.MlflowClient()

# create_table_statement = """
# drop table if exists dummy_metrics;
# create table dummy_metrics(
# 	timestamp timestamp,
# 	dataset_drift boolean,
#    share_of_drifted_columns float,
# 	prediction_drift float,
# 	num_drifted_columns integer,
# 	share_missing_values float
# )
# """

CREATE_TABLE_STATEMENT = """
drop table if exists dummy_metrics;
create table dummy_metrics(
    timestamp timestamp,
    prediction_drift float,
    num_drifted_columns integer,
    share_missing_values float
)
"""

begin = datetime.datetime(2022, 2, 1, 0, 0)
# column_mapping = ColumnMapping(
#     prediction='prediction',
#     numerical_features=num_features,
#     categorical_features=cat_features,
#     target=None
# )

# report = Report(metrics = [
#     ColumnDriftMetric(column_name='prediction'),
#     DatasetDriftMetric(),
#     DatasetMissingValuesMetric()
# ])
report = Report(
    metrics=[
        ColumnDriftMetric(column_name="prediction"),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        DataDriftPreset(),
        RegressionPreset(),
    ]
)


@task
def prep_db(connect_config, db_name):
    with psycopg.connect(connect_config, autocommit=True) as conn:
        res = conn.execute(
            f"SELECT 1 FROM pg_database WHERE datname='{db_name}'"
        )
        if len(res.fetchall()) == 0:
            conn.execute(f"create database {db_name};")
        with psycopg.connect(f"{connect_config} dbname={db_name}") as conn:
            conn.execute(CREATE_TABLE_STATEMENT)


@task
def calculate_metrics_postgresql(source_path, current_date, curr):

    current_path = f"{source_path}{current_date}"
    reference_date = pd.to_datetime(current_date) - pd.to_timedelta(
        7, unit="D"
    )
    reference_date = reference_date.strftime("%Y-%m-%d")
    reference_path = f"{source_path}{reference_date}"

    # Load current and reference data
    current_data = pd.read_csv(f"{current_path}/x_val.csv").drop(
        "order_purchase_date", axis=1
    )
    reference_data = pd.read_csv(f"{reference_path}/x_val.csv").drop(
        "order_purchase_date", axis=1
    )

    # Load model
    model = mlflow.catboost.load_model("models:/ecommerce_forecast/1")

    # Predictions
    reference_data["prediction"] = model.predict(
        reference_data
    )  # pd.read_csv(f"{reference_path}/y_val.csv")
    current_data["prediction"] = model.predict(current_data)

    # Prepare data to report
    reference_data["target"] = pd.read_csv(f"{reference_path}/y_val.csv")
    current_data["target"] = pd.read_csv(f"{current_path}/y_val.csv")

    report.run(
        reference_data=reference_data,
        current_data=current_data,
        # , column_mapping=column_mapping
    )

    result = report.as_dict()
    prediction_drift = result["metrics"][0]["result"]["drift_score"]
    num_drifted_columns = result["metrics"][1]["result"][
        "number_of_drifted_columns"
    ]
    share_missing_values = result["metrics"][2]["result"]["current"][
        "share_of_missing_values"
    ]
    # dataset_drift = result['metrics'][0]['result']["dataset_drift"]
    # result["metrics"][0]["result"]["share_of_drifted_columns"]
    curr.execute(
        # (timestamp, dataset_drift, share_of_drifted_columns)",
        "insert into "
        "dummy_metrics(timestamp, prediction_drift, "
        "num_drifted_columns, share_missing_values) "
        "values (%s, %s, %s, %s)",
        (
            current_date,
            prediction_drift,
            num_drifted_columns,
            share_missing_values,
        ),
    )

    # Save report
    report.save_html("reports/evidently_report.html")
    # pylint: disable=c-extension-no-member
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ecommerce_forecast_reports")
    with mlflow.start_run(run_name=f"catboost_report_{current_date}"):
        mlflow.log_artifact("reports/evidently_report.html")


# @flow
@click.command()
@click.option(
    "--source-path",
    default="./data/processed/",
    help="Location where the processed datasets were saved",
)
@click.option(
    "--current-date",
    default="2018-05-01",
    help="Current date prediction",
)
def batch_monitoring_backfill(source_path, current_date):

    password = os.getenv("POSTGRES_PASSWORD")
    connect_config = (
        f"host=localhost port=5432 user=postgres password={password}"
    )
    db_name = os.getenv("DB_NAME")

    prep_db(connect_config, db_name)
    with psycopg.connect(
        f"{connect_config} dbname={db_name}", autocommit=True
    ) as conn:
        with conn.cursor() as curr:
            calculate_metrics_postgresql(
                source_path=source_path, current_date=current_date, curr=curr
            )
        logging.info("data sent")


if __name__ == "__main__":
    batch_monitoring_backfill()
