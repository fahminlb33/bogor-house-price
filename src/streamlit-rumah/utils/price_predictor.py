import pandas as pd
import streamlit as st

import numpy as np
import pandas as pd
import lightgbm as lgb


@st.cache_resource
def load_model():
    model = lgb.Booster(model_file=st.secrets["REGRESSOR_PATH"])

    return model


def get_subdistricts():
    model = load_model()
    return model.pandas_categorical[0]


def predict_price(input_features: dict) -> pd.DataFrame:
    model = load_model()

    valid_locations = model.pandas_categorical[0]
    if input_features["subdistrict"][0] not in valid_locations:
        raise ValueError("Invalid subdistrict")

    df_predict = pd.DataFrame(input_features)
    df_predict["subdistrict"] = df_predict["subdistrict"].astype(
        pd.api.types.CategoricalDtype(valid_locations)
    )

    predicted = model.predict(df_predict)
    return float(np.exp(predicted[0]))
