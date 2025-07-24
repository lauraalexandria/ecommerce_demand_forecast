import logging

import click
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.metrics import mean_squared_error

logging.basicConfig(
    level=logging.INFO,
    filename="app.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ecommerce_forecast")


def create_mape_chart_by_date(forecast_df):

    forecast = forecast_df.copy().sort_values("month_min")
    forecast["mape"] = np.abs(
        (forecast_df["actual_value"] - forecast_df["forecast"])
        / forecast_df["actual_value"]
    )
    forecast["historical_mape"] = forecast.groupby(
        ["product_category_name", "customer_city"]
    )["mape"].cumsum()
    forecast["historical_mape"] = forecast["historical_mape"] / (
        forecast.groupby(
            ["product_category_name", "customer_city"]
        ).cumcount()
        + 1
    )

    return sns.relplot(
        data=forecast,
        x="month_min",
        y="historical_mape",
        kind="line",
        # hue = "method",
        row="customer_city",
        col="product_category_name",
        aspect=3,
    )


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

    logging.info("Loading datasets")
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
        with mlflow.start_run(run_name="catboost_tunning"):

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
            mape = (
                np.mean(
                    np.abs(
                        (np.array(y_val) - np.array(y_pred)) / np.array(y_val)
                    )
                )
                * 100
            )
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mape", mape)

            for prod in list(set(x_val["product_category_name"])):
                y_val_aux = y_val[
                    x_val[: len(y_val)]["product_category_name"] == prod
                ]
                y_pred_aux = y_pred[
                    x_val[: len(y_val)]["product_category_name"] == prod
                ]
                rmse = np.sqrt(mean_squared_error(y_val_aux, y_pred_aux))
                mlflow.log_metric(f"rmse_{prod}", rmse)
                mape = (
                    np.mean(
                        np.abs(
                            (np.array(y_val_aux) - np.array(y_pred_aux))
                            / np.array(y_val_aux)
                        )
                    )
                    * 100
                )
                mlflow.log_metric(f"mape_{prod}", mape)

            x_val_aux = x_val[: len(y_val)][
                ["month_min", "customer_city", "product_category_name"]
            ]
            x_val_aux["actual_value"] = y_val
            x_val_aux["forecast"] = y_pred

            ax = create_mape_chart_by_date(x_val_aux)
            # pylint: disable=protected-access
            fig = ax._figure
            mlflow.log_figure(fig, "historical_mape.png")
            plt.close(fig)

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
