from dataclasses import dataclass

from sqlalchemy import String, Integer, Float, text

from utils.db import db_session
from services import format_price, format_price_long


@dataclass
class District:
  district_sk: str
  district: str
  city: str


def list_districts():
  result = db_session.execute(
      text("SELECT * FROM marts_shared_dim_districts ORDER BY district ASC"))
  return [
      District(
          district_sk=row.district_sk, district=row.district, city=row.city)
      for row in result
  ]


def calc_median(table: str, column: str, where_clause: str = "") -> float:
  sql = f"""
  SELECT
    AVG(sub.{column}) AS median_col
  FROM ( 
    SELECT 
      p.{column},
      @row_index := @row_index + 1 AS row_index
    FROM
      {table} p, (SELECT @row_index := -1) r
    {where_clause}
    ORDER BY 
      p.{column} 
  ) AS sub
  WHERE 
    sub.row_index IN (FLOOR(@row_index / 2), CEIL(@row_index / 2))
  """

  return db_session.execute(text(sql)).scalar()


def calc_median_grouped(table: str,
                        column: str,
                        where_clause: str = "") -> list:
  sql = f"""
  WITH grouped_median AS (
    SELECT
      sub2.district_sk,
      sub2.total AS count_col,
      CASE
        WHEN MOD(sub2.total,2) = 1 THEN sub2.mid_prices
        WHEN MOD(sub2.total, 2) = 0 THEN (SUBSTRING_INDEX(sub2.mid_prices, ',', 1) + SUBSTRING_INDEX(sub2.mid_prices, ',', -1)) / 2
      END AS median_col
    FROM (
      SELECT
        sub1.district_sk,
        sub1.total,
        CASE
          WHEN MOD(sub1.total, 2) = 1 THEN SUBSTRING_INDEX(SUBSTRING_INDEX(sub1.prices, ',', CEIL(sub1.total / 2)), ',', '-1')
          WHEN MOD(sub1.total, 2) = 0 THEN SUBSTRING_INDEX(SUBSTRING_INDEX(sub1.prices, ',', sub1.total / 2 + 1), ',', '-2')
        END AS mid_prices
      FROM (
        SELECT
          district_sk,
          GROUP_CONCAT({column} ORDER BY {column}) AS prices,
          COUNT(*) AS total
        FROM
          {table}
        {where_clause}
        GROUP BY
          district_sk
      ) sub1
    ) sub2
  )
  SELECT 
    msdd.district_sk, 
    msdd.district, 
    CAST(grouped_median.median_col AS decimal) AS median_col,
    grouped_median.count_col
  FROM 
    grouped_median
  INNER JOIN 
    marts_shared_dim_districts msdd ON grouped_median.district_sk = msdd.district_sk 
  ORDER BY
    median_col DESC
  """

  return db_session.execute(text(sql)).fetchall()


@dataclass
class SummaryStatistics:
  top_district: str
  top_district_count: int
  median_price: str
  mean_price: str
  median_price_by_land_area: str
  median_price_by_building_area: str


def statistics(district_sk: str = "") -> SummaryStatistics:
  # load data from database
  where_clause = ""
  having_clause = ""
  if district_sk != "":
    where_clause = f"WHERE district_sk = '{district_sk}'"
    having_clause = f"HAVING district_sk = '{district_sk}'"

  # top district
  top_district = db_session.execute(
      text(f"""
  SELECT
    msdd.district_sk,
    msdd.district,
    COUNT(msdd.district) AS jumlah
  FROM
    marts_dashboard_fact_price mdfp 
  INNER JOIN
    marts_shared_dim_districts msdd 
    ON msdd.district_sk = mdfp.district_sk 
  GROUP BY 
    msdd.district_sk,
    msdd.district
  {having_clause}
  ORDER BY 
    jumlah DESC
  LIMIT 1""")).one()

  # price median and means
  price_median = calc_median("marts_dashboard_fact_price", "price",
                             where_clause)
  price_mean = db_session.execute(
      text(f"SELECT AVG(price) FROM marts_dashboard_fact_price {where_clause}")
  ).scalar()

  # median by land and building area
  land_median = calc_median("marts_dashboard_fact_price_ratio",
                            "price_per_land_area", where_clause)
  building_median = calc_median("marts_dashboard_fact_price_ratio",
                                "price_per_building_area", where_clause)

  return SummaryStatistics(
      top_district=top_district.district,
      top_district_count=int(top_district.jumlah),
      median_price=format_price_long(price_median),
      mean_price=format_price_long(price_mean),
      median_price_by_land_area=format_price_long(land_median),
      median_price_by_building_area=format_price_long(building_median),
  )


@dataclass
class HouseListingCount:
  district: str
  count: int


def house_listing_count() -> list[HouseListingCount]:
  result = db_session.execute(
      text(f"""
  SELECT 
    msdd.district,
    COUNT(mdfp.price) AS jumlah
  FROM
    marts_dashboard_fact_price mdfp 
  INNER JOIN
    marts_shared_dim_districts msdd ON msdd.district_sk = mdfp.district_sk 
  GROUP BY msdd.district DESC
  ORDER BY jumlah DESC
  """)).fetchall()
  return [
      HouseListingCount(district=row.district, count=int(row.jumlah))
      for row in result
  ]


@dataclass
class HousePriceMedian:
  district: str
  price_median: float
  count: int


def house_price_median() -> list[HousePriceMedian]:
  return [
      HousePriceMedian(
          district=row.district,
          price_median=float(row.median_col),
          count=int(row.count_col))
      for row in calc_median_grouped("marts_dashboard_fact_price", "price")
  ]


@dataclass
class HouseVariableCorrelation:
  variable: str
  method: str
  correlation: float
  p_value: float


def house_variable_correlation() -> list[HouseVariableCorrelation]:
  result = db_session.execute(
      text("SELECT * FROM marts_ml_correlations ORDER BY correlation DESC")
  ).fetchall()
  return [
      HouseVariableCorrelation(
          variable=row.variable,
          method=row.method,
          correlation=float(row.correlation),
          p_value=float(row.p_value)) for row in result
  ]


@dataclass
class HouseHistogramBar:
  bucket: float
  count: int


def house_histogram_bar(district_sk: str) -> list[HouseHistogramBar]:
  result = db_session.execute(
      text(f"""
    SELECT
      ROUND(price, -9) AS bucket,
      COUNT(*) AS count_col
    FROM (
      SELECT price
      FROM marts_dashboard_fact_price
      WHERE district_sk = :district_sk
    ) temp
    GROUP BY
      bucket"""), {
          "district_sk": district_sk
      }).fetchall()

  return [
      HouseHistogramBar(bucket=row.bucket, count=int(row.count_col))
      for row in result
  ]


@dataclass
class Amenity:
  amenity: str
  category: str
  lat: float
  lon: float


def list_amenities() -> list[Amenity]:
  result = db_session.execute(text("SELECT * FROM marts_spatial_amenities"))
  return [
      Amenity(
          amenity=row.amenity, category=row.category, lat=row.lat, lon=row.lon)
      for row in result
  ]
