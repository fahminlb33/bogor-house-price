import streamlit as st
import extra_streamlit_components as stx
import plotly.express as px

from utils.cookies import ensure_user_has_session
from utils.data_loaders import (formatter_pvalue, format_price, dim_districts,
                                fact_price, fact_listing_by_district,
                                fact_price_by_district, fact_correlations,
                                fact_price_ratio, load_geojson_bogor)


def derive_quick_stats():
  df_fact_price = fact_price()
  df_fact_price_ratio = fact_price_ratio()
  df_fact_listing_by_district = fact_listing_by_district()

  return {
      "top_district":
          df_fact_listing_by_district.sort_values(
              'listing_count', ascending=False).iloc[0, 1],
      "median_price":
          df_fact_price['price'].median(),
      "mean_price":
          df_fact_price['price'].mean(),
      "median_price_pl":
          df_fact_price_ratio["price_per_land_area"].median(),
      "median_price_pb":
          df_fact_price_ratio["price_per_building_area"].median(),
  }


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

  st.title("Dasbor Harga Rumah Bogor")

  # load dataset
  df_dim_districts = dim_districts()
  df_fact_price = fact_price()
  df_fact_correlations = fact_correlations()
  df_fact_price_by_district = fact_price_by_district()
  df_fact_listing_by_district = fact_listing_by_district()
  geojson_bogor = load_geojson_bogor()

  # ---- quick statistics
  quick_stats = derive_quick_stats()
  col1, col2, col3 = st.columns(3)
  col1.metric("Kecamatan dengan Listing Terbanyak", quick_stats["top_district"])
  col2.metric("Median Harga Rumah", format_price(quick_stats["median_price"]))
  col3.metric("Rata-Rata Harga Rumah", format_price(quick_stats["mean_price"]))

  col1, col2 = st.columns(2)
  col1.metric("Median Harga per Luas Tanah (m^2)",
              format_price(quick_stats["median_price_pl"]))
  col2.metric("Median Harga per Luas Bangunan (m^2)",
              format_price(quick_stats["median_price_pb"]))

  # ---- top 10 districts by listing count
  ch_price_df = df_fact_price_by_district \
      .sort_values("price_median", ascending=False) \
      .head(10)

  ch_price = px.bar(
      ch_price_df,
      x="district",
      y="price_median",
      title="10 Kecamatan dengan Harga Rumah Tertinggi",
      labels={
          "price_median": "Median Harga Rumah",
          "district": "Kecamatan"
      })
  st.plotly_chart(ch_price)

  # ---- top 10 districts by price box plot
  top_10_district_by_listing = df_fact_listing_by_district \
      .sort_values("listing_count", ascending=False) \
      .head(10).index
  ch_price_box_df = df_fact_price.loc[top_10_district_by_listing]

  ch_price_box = px.box(
      ch_price_box_df,
      x="district",
      y="price",
      title="Distribusi Harga Rumah di 10 Kecamatan dengan Listing Terbanyak",
      labels={
          "price": "Harga Rumah",
          "district": "Kecamatan"
      })
  st.plotly_chart(ch_price_box)

  # ---- correlation per variable to price
  st.subheader("Korelasi Harga Rumah dengan Variabel Lainnya")

  df_corr = df_fact_correlations.round(2).style \
      .background_gradient(cmap="coolwarm", subset=["correlation"]) \
      .map(formatter_pvalue, subset=["p_value"])

  st.dataframe(df_corr)
  st.caption("Merah = signifikan pada alpha < 0,05")

  # ---- number of listings per district in map

  # derive the map data
  df_choro = df_fact_price.merge(df_dim_districts, on="district_sk") \
      .groupby("district_x") \
      .agg({"price": "median", "district_y": "size"}) \
      .reset_index() \
      .rename(columns={"district_x": "district", "district_y": "count"})

  # create choropleth map
  ch_choro = px.choropleth_mapbox(
      df_choro,
      geojson=geojson_bogor,
      locations="district",
      color="count",
      featureidkey="properties.NAMOBJ",
      color_continuous_scale="Viridis",
      mapbox_style="carto-positron",
      zoom=8,
      opacity=0.5,
      center={
          "lat": -6.6,
          "lon": 106.8
      },
      hover_data={
          "district": True,
          "count": True,
          "price": ":,.0f"
      })
  ch_choro.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
  ch_choro.update_layout(coloraxis_showscale=False)

  # plot choropleth map
  st.subheader("Peta Listing Rumah per Kecamatan")
  st.plotly_chart(ch_choro)


if __name__ == "__main__":
  # bootstrap
  main()
