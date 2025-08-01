import logging

import pandas as pd
from workalendar.america import Brazil

logging.basicConfig(
    level=logging.INFO,
    filename="app.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def add_temporal_features(df, date_col):
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day_of_month"] = df[date_col].dt.day
    df["weekday"] = df[date_col].dt.day_name()

    return df


def add_holidays(df, cal):
    df = df.merge(
        pd.concat(
            [
                pd.DataFrame(
                    cal.holidays(2017),
                    columns=["order_purchase_date", "holiday"],
                ),
                pd.DataFrame(
                    cal.holidays(2018),
                    columns=["order_purchase_date", "holiday"],
                ),
            ]
        ).astype(dtype={"order_purchase_date": "datetime64[ns]"}),
        how="left",
    ).fillna({"holiday": "missing"})
    df["flag_holiday"] = (df["holiday"] != "missing").astype(int)

    return df


def detect_new_clients(df, key_col_list):
    df["flag_new_client"] = df.groupby(key_col_list).cumcount() + 1
    df["flag_new_client"] = (df["flag_new_client"] == 1).astype(int)

    return df


def aggregate_cols_by_dates(df, key_cols_list):
    df = df.copy()
    df = df.groupby(key_cols_list).agg(
        {
            "year": "min",
            "month": "min",
            "day_of_month": "max",
            "flag_holiday": "max",
            "flag_approved_order": "mean",
            "flag_new_client": "mean",
            "daytime_in_minutes": ["mean", "median", "min", "max"],
            "sales_amount": ["sum", "mean", "median", "min", "max"],
            "sales_value": ["sum", "mean", "median", "min", "max"],
            "freight": ["mean", "median", "min", "max"],
            "product_weight_g": ["mean", "median"],
        }
    )

    df.columns = df.columns.droplevel(1) + "_" + df.columns.droplevel(0)
    df = df.reset_index()

    return df


def filter_products_and_cities(df, prod_list, cities_list):
    return df[
        (df["product_category_name"].isin(prod_list))
        & (df["customer_city"].isin(cities_list))
    ]


def avoid_gap_dates(df, date_col, key_cols_list):
    start_date = df[date_col].min()
    final_date = df[date_col].max()
    logging.info(final_date - start_date)

    dates_df = pd.DataFrame(
        {date_col: pd.date_range(start=start_date, end=final_date, freq="D")}
    )

    dates_df = dates_df.merge(
        df[key_cols_list].drop_duplicates(), how="cross"
    )

    return df.merge(dates_df, how="right").fillna(
        value={
            "sales_amount_sum": 0,
            "sales_amount_mean": 0,
            "sales_amount_median": 0,
            "sales_amount_min": 0,
            "sales_amount_max": 0,
            "sales_value_sum": 0,
            "sales_value_mean": 0,
            "sales_value_median": 0,
            "sales_value_min": 0,
            "sales_value_max": 0,
        }
    )


if __name__ == "__main__":
    logging.info("Reading datasets")
    orders_df = pd.read_csv("./data/raw/olist_orders_dataset.csv")
    order_items_df = pd.read_csv("./data/raw/olist_order_items_dataset.csv")
    products_df = pd.read_csv("./data/raw/olist_products_dataset.csv")
    customers_df = pd.read_csv("./data/raw/olist_customers_dataset.csv")

    order_items_df = (
        order_items_df.groupby(["order_id", "product_id"])
        .agg(
            sales_amount=("price", "count"),
            sales_value=("price", "sum"),
            freight=("freight_value", "sum"),
        )
        .reset_index()
    )

    logging.info("Join initial datasets")
    joined_df = (
        orders_df.merge(order_items_df, how="left", on=["order_id"])
        .merge(products_df, how="left", on=["product_id"])
        .merge(customers_df, how="left", on=["customer_id"])
    )

    logging.info("Improve date features")
    joined_df["order_purchase_timestamp"] = pd.to_datetime(
        joined_df["order_purchase_timestamp"]
    )
    joined_df["order_purchase_date"] = [
        pd.to_datetime(day) - pd.Timedelta(days=day.weekday())
        for day in joined_df["order_purchase_timestamp"].dt.normalize()
    ]
    joined_df["order_purchase_original_date"] = pd.to_datetime(
        joined_df["order_purchase_timestamp"]
    ).dt.normalize()
    joined_df["order_purchase_month"] = pd.to_datetime(
        pd.to_datetime(joined_df["order_purchase_timestamp"]).dt.strftime(
            "%Y-%m-01"
        )
    )
    joined_df["daytime_in_minutes"] = (
        pd.to_datetime(joined_df["order_purchase_timestamp"]).dt.hour * 60
        + pd.to_datetime(joined_df["order_purchase_timestamp"]).dt.minute
    )
    joined_df = joined_df.sort_values("order_purchase_original_date")

    logging.info("Filter dataset")
    selected_prod_list = [
        "cama_mesa_banho",
        "beleza_saude",
        "esporte_lazer",
        "informatica_acessorios",
        "moveis_decoracao",
    ]
    selected_cities_list = ["sao paulo"]

    joined_df = joined_df[
        (joined_df["order_purchase_month"] >= pd.to_datetime("2017-01-01"))
        & (joined_df["order_purchase_month"] <= pd.to_datetime("2018-08-01"))
    ]

    logging.info("Add new features")
    joined_df["flag_approved_order"] = (
        ~joined_df["order_status"].isin(["unavailable", "canceled"])
    ).astype(int)
    joined_df = add_temporal_features(
        joined_df, date_col="order_purchase_original_date"
    )
    brazilian_cal = Brazil()
    joined_df = add_holidays(joined_df, cal=brazilian_cal)
    joined_df = detect_new_clients(
        joined_df,
        [
            "customer_id",
            "product_category_name",
            "customer_state",
            "customer_city",
        ],
    )

    logging.info("Aggregate dataset weekly")
    national_df = aggregate_cols_by_dates(
        joined_df, ["order_purchase_date", "product_category_name"]
    )
    joined_df = filter_products_and_cities(
        joined_df,
        prod_list=selected_prod_list,
        cities_list=selected_cities_list,
    )
    joined_df = avoid_gap_dates(
        joined_df,
        "order_purchase_original_date",
        ["product_category_name", "customer_state", "customer_city"],
    )
    final_df = aggregate_cols_by_dates(
        joined_df,
        [
            "order_purchase_date",
            "product_category_name",
            "customer_state",
            "customer_city",
        ],
    )

    logging.info("Write datasets")
    national_df.to_csv(
        "./data/processed/national_orders_by_week.csv", index=False
    )
    final_df.to_csv("./data/processed/orders_by_week.csv", index=False)
