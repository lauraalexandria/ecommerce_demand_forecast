import pandas as pd

from scr.model_pipeline.data_preparation import aggregate_cols_by_dates


def test_prepare_data():

    data = [
        (2017, 5, 20, 0, 1, 1, 1000, 1, 75, 5, 30),
        (2017, 5, 20, 0, 1, 1, 500, 1, 150, 5, 100),
        (2017, 5, 20, 0, 1, 1, 800, 1, 100, 5, 30),
        (2017, 5, 20, 0, 1, 1, 300, 1, 50, 5, 55),
    ]

    columns = [
        "year",
        "month",
        "day_of_month",
        "flag_holiday",
        "flag_approved_order",
        "flag_new_client",
        "daytime_in_minutes",
        "sales_amount",
        "sales_value",
        "freight",
        "product_weight_g",
    ]
    df = pd.DataFrame(data, columns=columns)

    new_columns = [
        "year",
        "month",
        "day_of_month",
        "flag_holiday",
        "flag_approved_order",
        "flag_new_client",
        "daytime_in_minutes_mean",
        "daytime_in_minutes_median",
        "daytime_in_minutes_min",
        "daytime_in_minutes_max",
        "sales_amount_sum",
        "sales_amount_mean",
        "sales_amount_median",
        "sales_amount_min",
        "sales_amount_max",
        "sales_value_sum",
        "sales_value_mean",
        "sales_value_median",
        "sales_value_min",
        "sales_value_max",
        "freight_mean",
        "freight_median",
        "freight_min",
        "freight_max",
        "product_weight_g_mean",
        "product_weight_g_median",
    ]

    actual_df = aggregate_cols_by_dates(df, key_cols_list=[])
    expected_df = pd.DataFrame(
        [
            (
                2017,
                5,
                20,
                0,
                1,
                1,
                650,
                650,
                300,
                1000,
                4,
                1,
                1,
                1,
                375,
                93.75,
                87.5,
                50,
                150,
                5,
                5,
                5,
                5,
                53.75,
                42.5,
            )
        ],
        columns=new_columns,
    )

    assert all(actual_df == expected_df)
