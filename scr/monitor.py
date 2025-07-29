import click
import mlflow
import pandas as pd
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.report import Report

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

    # Create and save report
    report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html("reports/evidently_report.html")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ecommerce_forecast_reports")
    with mlflow.start_run(run_name=f"catboost_report_{current_date}"):
        mlflow.log_artifact("reports/evidently_report.html")


if __name__ == "__main__":
    run_monitor()
