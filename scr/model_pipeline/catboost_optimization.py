import logging

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.metrics import mean_squared_error

import mlflow

logging.basicConfig(
    level=logging.INFO,
    filename="app.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ecommerce_forecast")


def latest_value_forecast(
    df, group_col, value_col, date_col, days_to_predict
):
    """
    Forecast by replicating the last value for each group
    Returns DataFrame with forecast dates and group values
    """

    df = df.copy()
    df[date_col] = df[date_col] + pd.to_timedelta(days_to_predict, unit="D")
    df["forecast"] = df[value_col]
    df["method"] = "latest_value_forecast"

    return df[[date_col, *group_col, value_col, "forecast", "method"]]


# pylint: disable=too-many-arguments, too-many-positional-arguments
def moving_average_forecast(
    df, group_col, value_col, date_col, days_to_predict, window
):
    """
    Forecast using moving average for each group
    Returns DataFrame with forecast dates and group values
    """

    df = df.copy()
    df[date_col] = df[date_col] + pd.to_timedelta(days_to_predict, unit="D")
    for i in range(window):
        df[f"lag{i}"] = df.groupby(group_col)[value_col].shift(i)

    lags = [f"lag{i}" for i in range(window)]
    df["forecast"] = df[lags].mean(axis=1)
    df["method"] = "ma_forecast"

    return df[[date_col, *group_col, value_col, "forecast", "method"]]


def calculate_mape(df):
    return (df["actual_value"] - df["forecast"]) / df["actual_value"]


def create_mape_chart_by_date(df):

    df["mape"] = list(
        df.groupby(
            [
                "order_purchase_date",
                "product_category_name",
                "customer_city",
                "method",
            ]
        ).apply(calculate_mape)
    )

    return sns.relplot(
        data=df,
        x="order_purchase_date",  # "month_min",
        y="mape",  # "historical_mape",
        kind="line",
        hue="method",
        row="customer_city",
        col="product_category_name",
        aspect=1,
    )


# pylint: disable=too-many-locals
@click.command()
@click.option(
    "--source-path",
    default="./data/processed/",
    help="Location where the processed datasets were saved",
)
@click.option(
    "--split-data",
    default="2018-05-01",
    help="Split date between train/test datasets. First date in test file",
)
@click.option(
    "--num_trials",
    default=15,
    help="The number of parameter evaluations for the optimizer to explore",
)
def run_optimization(source_path: str, split_data: str, num_trials: int):

    source_path = f"{source_path}{split_data}/"
    logging.info("Loading datasets")
    x_train = pd.read_csv(f"{source_path}x_train.csv").drop(
        "order_purchase_date", axis=1
    )
    y_train = pd.read_csv(f"{source_path}y_train.csv")
    x_val = pd.read_csv(f"{source_path}x_val.csv").drop(
        "order_purchase_date", axis=1
    )
    dates_val = pd.to_datetime(
        pd.read_csv(f"{source_path}x_val.csv")["order_purchase_date"]
    )
    y_val = pd.read_csv(f"{source_path}y_val.csv")
    cat_cols = list(x_train.select_dtypes("object").columns)

    def objective(params):

        mlflow.autolog()
        with mlflow.start_run(run_name=f"catboost_tunning_{split_data}"):

            mlflow.log_params(params)
            model = CatBoostRegressor(
                random_seed=56, cat_features=cat_cols, verbose=0
            )
            model.fit(x_train, y_train)

            mlflow.catboost.log_model(model, name="model")
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

            target_col = "actual_value"  # f"target_{horizon}_semana"
            group_col = ["product_category_name", "customer_city"]
            date_col = "order_purchase_date"
            days = 7
            window = 3

            x_val_aux = x_val[: len(y_val)][[*group_col]]
            x_val_aux["actual_value"] = y_val
            x_val_aux["forecast"] = y_pred
            x_val_aux["order_purchase_date"] = dates_val[
                : len(y_val)
            ] + pd.to_timedelta(days, unit="D")
            x_val_aux["method"] = "forecast"

            forecast_df = pd.concat(
                [
                    latest_value_forecast(
                        x_val_aux, group_col, target_col, date_col, days
                    ),
                    moving_average_forecast(
                        x_val_aux,
                        group_col,
                        target_col,
                        date_col,
                        days,
                        window,
                    ),
                    x_val_aux,
                ]
            )  # .drop(target_col, axis=1)

            ax = create_mape_chart_by_date(forecast_df)
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
