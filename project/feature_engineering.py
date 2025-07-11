import pandas as pd


def add_tendency_features(df, feat_list, key_col_list):
    for col in feat_list:
        df[f"{col}_lag"] = df.groupby(key_col_list)[col].shift(1)
        df[f"{col}_lag2"] = df.groupby(key_col_list)[col].shift(2)
        df[f"{col}_lag3"] = df.groupby(key_col_list)[col].shift(3)
        df[f"{col}_lag4"] = df.groupby(key_col_list)[col].shift(4)
        df[f"{col}_lag12"] = df.groupby(key_col_list)[col].shift(12)

        df[f"{col}_historical_mean"] = df.groupby(key_col_list)[col].cumsum()
        df[f"{col}_historical_mean"] = df[f"{col}_historical_mean"] / (
            df.groupby(key_col_list).cumcount() + 1
        )
        df[f"{col}_historical_diff"] = (
            df[col] - df[f"{col}_historical_mean"]
        ) / df[f"{col}_historical_mean"]

    return df


if __name__ == "__main__":
    print("Reading datasets")
    final_df = pd.read_csv("./data/processed/orders_by_week.csv")
    national_df = pd.read_csv("./data/processed/national_orders_by_week.csv")

    selected_feat_list = [
        "flag_approved_order_mean",
        "daytime_in_minutes_mean",
        "sales_amount_mean",
        "sales_amount_sum",
        "sales_value_sum",
        "frete_total_mean",
        "product_weight_g_mean",
        "flag_approved_order_mean",
        "flag_approved_order_mean_national",
        "flag_new_client_mean",
        "flag_new_client_mean_national",
        "daytime_in_minutes_mean_national",
        "sales_amount_mean_national",
        "frete_total_mean_national",
        "product_weight_g_mean_national",
    ]

    print("Add historical features")
    final_df = final_df.sort_values("order_purchase_date")
    final_df = final_df.merge(
        national_df,
        how="left",
        on=["order_purchase_date", "product_category_name"],
        suffixes=("", "_national"),
    )
    final_df = add_tendency_features(
        final_df,
        feat_list=selected_feat_list,
        key_col_list=["product_category_name", "customer_city"],
    )

    print("Write datasets")
    final_df.to_csv("./data/processed/model_data.csv", index=False)
