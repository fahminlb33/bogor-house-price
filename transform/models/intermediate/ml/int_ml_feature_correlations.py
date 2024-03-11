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
    dbt.config(materialized="table")

    # load clean dataset
    df = dbt.ref("int_ml_feature_outlier_removal")

    # calculate point biserial/pearson correlation for each column
    corrs = []
    for column in df.columns:
        # skip price column
        if column == "price":
            continue

        # get the data
        data = df.select(column, "price").fetchdf()

        # check if column is numeric
        if not np.issubdtype(data[column].dtype, np.number):
            continue

        # calculate correlation
        if should_use_point_biser_corr(column):
            method = "pointbiser"
            corr = stats.pointbiserialr(data[column], data["price"])
        else:
            method = "pearson"
            corr = stats.pearsonr(data[column], data["price"])

        # append to corrs
        corrs.append({
            "variable": column,
            "method": method,
            "correlation": corr[0],
            "p_value": corr[1]
        })

    return pd.DataFrame(corrs)
