import time

import click
import pandas as pd
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.report import Report
from prometheus_client import Gauge, start_http_server

import mlflow

# Initiate Prometheus server in port
start_http_server(3030)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.MlflowClient()


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
def run_monitor(source_path: str, current_date: str):

    # Cria métricas Prometheus
    drift_status = Gauge(
        "evidently_dataset_drift", "Status de Drift (0 ou 1)"
    )
    drift_share = Gauge(
        "evidently_drift_share", "Percentual de Features com Drift"
    )

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
    print(current_data.head())

    def calculate_metrics(reference_df, current_df):

        report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
        report.run(reference_data=reference_df, current_data=current_df)

        results = report.as_dict()

        # Extrai métricas
        drift_status.set(
            1 if results["metrics"][0]["result"]["dataset_drift"] else 0
        )
        drift_share.set(
            results["metrics"][0]["result"]["share_of_drifted_columns"] * 100
        )

    # Executa a cada hora (simplificado)
    while True:
        calculate_metrics(reference_data, current_data)
        time.sleep(3600)


if __name__ == "__main__":
    run_monitor()
