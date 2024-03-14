import json

import pandas as pd
import streamlit as st

from babel.numbers import format_compact_currency, format_currency


def formatter_pvalue(x):
    return "background-color: red" if x < 0.05 else None


def format_price(x):
    return format_compact_currency(x, "IDR", locale="id_ID")


def format_price_long(x):
    return format_currency(x, "IDR", locale="id_ID")


def percent_change(x, y):
    return (x - y) / y


@st.cache_data()
def dim_districts():
    return pd.read_csv("assets/data/marts_shared_dim_districts.csv").set_index(
        "district_sk")


@st.cache_data()
def fact_price_by_district():
    return pd.read_csv("assets/data/marts_dashboard_fact_price_by_district.csv") \
        .merge(dim_districts(), on="district_sk") \
        .set_index("district_sk")


@st.cache_data()
def fact_listing_by_district():
    return pd.read_csv("assets/data/marts_dashboard_fact_listing_by_district.csv") \
        .merge(dim_districts(), on="district_sk") \
        .set_index("district_sk")


@st.cache_data()
def fact_price():
    return pd.read_csv("assets/data/marts_dashboard_fact_price.csv") \
        .merge(dim_districts(), on="district_sk") \
        .set_index("district_sk")


@st.cache_data()
def fact_correlations():
    return pd.read_csv("assets/data/marts_ml_correlations.csv")


@st.cache_data()
def fact_price_ratio():
    return pd.read_csv("assets/data/marts_dashboard_fact_price_ratio.csv") \
        .merge(dim_districts(), on="district_sk") \
        .set_index("district_sk")


@st.cache_data()
def load_geojson_bogor():
    with open("assets/data/bogor.json") as f:
        return json.load(f)
