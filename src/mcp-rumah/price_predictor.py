import pandas as pd

import numpy as np
import pandas as pd
import lightgbm as lgb

from config import config

model = lgb.Booster(model_file=config("REGRESSOR_PATH"))


def get_subdistricts():
    return model.pandas_categorical[0]


def predict_price(input_features: dict) -> pd.DataFrame:
    valid_locations = model.pandas_categorical[0]
    if input_features["subdistrict"][0] not in valid_locations:
        raise ValueError("Invalid subdistrict")

    df_predict = pd.DataFrame(input_features)
    df_predict["subdistrict"] = df_predict["subdistrict"].astype(
        pd.api.types.CategoricalDtype(valid_locations)
    )

    predicted = model.predict(df_predict)
    return float(np.exp(predicted[0]))
