import streamlit as st
import extra_streamlit_components as stx

import plotly.express as px

from babel.numbers import format_percent
from statsmodels.stats.descriptivestats import describe

from utils.cookies import ensure_user_has_session
from utils.data_loaders import (format_price, percent_change, dim_districts,
                                fact_price, fact_listing_by_district,
                                fact_price_by_district, fact_price_ratio)


def main():
  #
  # Page configuration
  #

  # set page config
  st.set_page_config(
      page_title="Dasbor - NyariRumah",
      page_icon="👋",
  )

  # set cookie manager
  cookie_manager = stx.CookieManager()
  ensure_user_has_session(cookie_manager)

  #
  # Page contents
  #

  # load data
  df_dim_districts = dim_districts().sort_values("district", ascending=True)
  df_fact_price = fact_price()
  df_fact_price_ratio = fact_price_ratio()
  df_fact_price_by_district = fact_price_by_district()
  df_fact_listing_by_district = fact_listing_by_district()

  # global stats
  global_median = df_fact_price["price"].median()
  global_avg = df_fact_price["price"].mean()

  global_land_median = df_fact_price_ratio["price_per_land_area"].median()
  global_building_median = df_fact_price_ratio[
      "price_per_building_area"].median()

  # populate sidebar
  with st.sidebar:
    district_sk = st.selectbox(
        "Pilih kota/kecamatan",
        df_dim_districts.index,
        format_func=lambda x: df_dim_districts.loc[x, "district"])

  # set title
  st.title("Statistik Kecamatan " +
           df_dim_districts.loc[district_sk, "district"])

  # quick statistics
  col1, col2, col3 = st.columns(3)
  # --- total listing
  col1.metric("Total Listing", df_fact_listing_by_district.loc[district_sk,
                                                               "listing_count"])
  # --- median price
  district_median = df_fact_price_by_district.loc[district_sk, "price_median"]
  col2.metric(
      "Median Harga Rumah",
      format_price(district_median),
      format_percent(percent_change(district_median, global_median)),
      delta_color="inverse")
  # --- average price
  district_avg = df_fact_price_by_district.loc[district_sk, "price_avg"]
  col3.metric(
      "Rata-Rata Harga Rumah",
      format_price(district_avg),
      format_percent(percent_change(district_avg, global_avg)),
      delta_color="inverse")

  col1, col2 = st.columns(2)
  # --- median price per land area
  land_median = df_fact_price_ratio.loc[district_sk][
      "price_per_land_area"].median()
  col1.metric(
      "Median Harga per Luas Tanah (m^2)",
      format_price(land_median),
      format_percent(percent_change(land_median, global_land_median)),
      delta_color="inverse")
  # --- median price per building area
  building_median = df_fact_price_ratio.loc[district_sk][
      "price_per_building_area"].median()
  col2.metric(
      "Median Harga per Luas Bangunan (m^2)",
      format_price(building_median),
      format_percent(percent_change(building_median, global_building_median)),
      delta_color="inverse")

  st.caption("Perbandingan dengan median dan rata-rata kota")

  # price histogram
  ch_price_hist = px.histogram(
      df_fact_price.loc[district_sk],
      x="price",
      nbins=20,
      marginal="box",
      title="Distribusi Harga Rumah",
      labels={"price": "Harga Rumah"})
  st.plotly_chart(ch_price_hist)

  # descriptive
  st.subheader("Statistik Deskriptif Harga Rumah")
  df_stats = describe(df_fact_price.loc[district_sk]["price"])
  # split rows into two columns
  col1, col2 = st.columns(2)
  with col1:
    st.table(df_stats.loc[[
        "min", "mean", "median", "mode", "std", "std_err", "max", "25%", "50%",
        "75%"
    ]])
  with col2:
    st.table(df_stats.loc[[
        "iqr", "coef_var", "skew", "kurtosis", "upper_ci", "lower_ci",
        "jarque_bera", "jarque_bera_pval"
    ]])


if __name__ == "__main__":
  # bootstrap
  main()
