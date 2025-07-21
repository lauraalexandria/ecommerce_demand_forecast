import mlflow
import logging

logging.basicConfig(
    level=logging.INFO,
    filename='app.log',
    format='%(asctime)s - %(levelname)s - %(message)s'
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
        f"Melhor run_id: {best_run.info.run_id}",
        f"RMSE: {best_run.data.metrics['rmse']}",
    )

    mlflow.artifacts.download_artifacts(
        best_model_uri,
        dst_path="final_model",
    )


if __name__ == "__main__":
    load_best_model()
