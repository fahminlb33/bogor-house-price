import numpy as np
import pandas as pd
import scipy.stats as stats

IGNORE_COLUMNS = [
    "id", "price", "installment", "district", "city", "description", "url",
    "last_modified", "scraped_at", "place"
]


def should_use_point_biser_corr(column):
    return "tag_" in column or "facility_" in column or "_mat_" in column


def model(dbt, session):
    # load clean dataset
    df = dbt.ref("int_ml_feature_outlier_removal")

    # get the price field
    price_col = np.log(list(map(lambda x: x[0], df.select("price").fetchall())))

    # calculate point biserial/pearson correlation for each column
    corrs = []
    for column in df.columns:
        # skip price column
        if column == "price":
            continue

        # get the data
        data = list(map(lambda x: x[0], df.select(column).fetchall()))

        # check if column is numeric
        if not pd.api.types.is_number(data[0]):
            continue

        # calculate correlation
        if should_use_point_biser_corr(column):
            method = "pointbiser"
            corr = stats.pointbiserialr(data, price_col)
        else:
            method = "pearson"
            corr = stats.pearsonr(data, price_col)

        # append to corrs
        corrs.append({
            "variable": column,
            "method": method,
            "correlation": corr[0],
            "p_value": corr[1]
        })

    # calculate pearson correlation for spatial data
    df_spatial = dbt.ref("marts_spatial_price").df()
    for col in df_spatial.columns:
        if col in IGNORE_COLUMNS:
            continue

        # calculate correlation
        corr = stats.pearsonr(df_spatial[col], df_spatial["price"])
        corrs.append({
            "variable": f"spatial_{col}",
            "method": "pearson",
            "correlation": corr[0],
            "p_value": corr[1]
        })

    return pd.DataFrame(corrs)
