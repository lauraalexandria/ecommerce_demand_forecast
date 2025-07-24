import datetime

import mlflow
import pandas as pd
from catboost import CatBoost
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.report import Report

# Load current and reference data
current_data = pd.read_csv("./data/processed/x_val.csv").drop(
    "order_purchase_date", axis=1
)

# Load model
model = CatBoost()
model.load_model("final_model/model/model.cb")
model.set_feature_names(current_data.columns)

# Predictions
current_data["prediction"] = model.predict(current_data)

# Prepare data to report
current_data["target"] = pd.read_csv("./data/processed/y_val.csv")
reference_data = current_data.copy()
reference_data["prediction"] = pd.read_csv("./data/processed/y_val.csv")

# Create and save report
report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
report.run(reference_data=reference_data, current_data=current_data)
report.save_html("reports/evidently_report.html")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ecommerce_forecast_reports")
with mlflow.start_run(run_name=f"catboost_report_{datetime.date.today()}"):
    mlflow.log_artifact("reports/evidently_report.html")
