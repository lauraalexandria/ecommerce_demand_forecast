import click
import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ecommerce_forecast")


@click.command()
@click.option(
    "--source-path",
    default="./data/processed/",
    help="Location where the processed datasets were saved",
)
@click.option(
    "--num_trials",
    default=15,
    help="The number of parameter evaluations for the optimizer to explore",
)
def run_optimization(source_path: str, num_trials: int):

    # pylint: disable=duplicate-code
    print("Loading datasets")
    x_train = pd.read_csv(f"{source_path}x_train.csv").drop(
        "order_purchase_date", axis=1
    )
    y_train = pd.read_csv(f"{source_path}y_train.csv")
    x_val = pd.read_csv(f"{source_path}x_val.csv").drop(
        "order_purchase_date", axis=1
    )
    y_val = pd.read_csv(f"{source_path}y_val.csv")
    cat_cols = list(x_train.select_dtypes("object").columns)

    def objective(params):

        mlflow.autolog()
        with mlflow.start_run():

            mlflow.log_params(params)
            model = CatBoostRegressor(
                random_seed=56, cat_features=cat_cols, verbose=0
            )
            model.fit(x_train, y_train)

            mlflow.catboost.log_model(model, "model")
            y_pred = model.predict(x_val).round()

            y_val.dropna(inplace=True)
            y_pred = y_pred[: len(y_val)]
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            # mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
            mlflow.log_metric("rmse", rmse)
            # mlflow.log_metric("mape", mape)

        return {"loss": rmse, "status": STATUS_OK}

    search_space = {
        "depth": scope.int(hp.quniform("depth", 1, 20, 1)),
        "iterations": scope.int(hp.quniform("iterations", 10, 50, 1)),
        "min_data_in_leaf": scope.int(
            hp.quniform("min_data_in_leaf", 1, 4, 1)
        ),
        "random_state": 42,
    }

    rstate = np.random.default_rng(42)
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate,
    )


if __name__ == "__main__":
    run_optimization()
