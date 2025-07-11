from datetime import timedelta

import click
import pandas as pd


@click.command()
@click.option(
    "--input-path",
    default="./data/processed/model_data.csv",
    help="Path to the data",
)
@click.option(
    "--target-col-source",
    default="sales_amount_sum",
    help="Column name used as the future target",
)
@click.option(
    "--horizon",
    default=1,
    help="Number of weeks in the future to set the target",
)
@click.option(
    "--split-data",
    default="2018-05-01",
    help="Split date between train/test datasets. First date in test file",
)
def add_target_and_split_by_product(
    input_path, target_col_source, horizon, split_data
):

    df = pd.read_csv(input_path)
    df["order_purchase_date"] = pd.to_datetime(df["order_purchase_date"])
    df = df.sort_values("order_purchase_date")

    df[f"target_{horizon}_semana"] = df.groupby(
        ["product_category_name", "customer_city"]
    )[target_col_source].shift(-horizon)

    split_data = pd.to_datetime("2018-05-01")
    df_train = df[df["order_purchase_date"] < split_data]
    df_val = df[df["order_purchase_date"] >= split_data + timedelta(days=7)]

    x_train = df_train.drop(f"target_{horizon}_semana", axis=1)
    y_train = df_train[f"target_{horizon}_semana"]
    x_val = df_val.drop(f"target_{horizon}_semana", axis=1)
    y_val = df_val[f"target_{horizon}_semana"]

    x_train.to_csv("./data/processed/x_train.csv", index=False)
    y_train.to_csv("./data/processed/y_train.csv", index=False)
    x_val.to_csv("./data/processed/x_val.csv", index=False)
    y_val.to_csv("./data/processed/y_val.csv", index=False)


if __name__ == "__main__":
    add_target_and_split_by_product()
