import logging

import mlflow

logging.basicConfig(
    level=logging.INFO,
    filename="app.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

EXPERIMENT_NAME = "ecommerce_forecast"
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.MlflowClient()


def load_best_model():

    logging.info("Load best model")
    experiment_id = client.get_experiment_by_name(
        EXPERIMENT_NAME
    ).experiment_id
    runs = client.search_runs(
        experiment_id, order_by=["metrics.rmse ASC"], max_results=1
    )

    best_run = runs[0]
    best_model_uri = f"runs:/{best_run.info.run_id}/model"

    logging.info(
        "Best run_id: %s, RMSE: %s",
        best_run.info.run_id,
        best_run.data.metrics["rmse"],
    )

    mlflow.artifacts.download_artifacts(
        best_model_uri,
        dst_path="final_model",
    )

    # Registrando no MLFlow, posso apagar a parte de baixar localmente?
    mlflow.register_model(best_model_uri, "ecommerce_forecast")

    client.transition_model_version_stage(
        name="ecommerce_forecast", version=1, stage="Production"
    )


if __name__ == "__main__":
    load_best_model()
