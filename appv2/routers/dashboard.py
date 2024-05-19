import dataclasses

from flask import Blueprint, render_template

from utils.shared import cache
from services.repository.data_mart import (statistics,
                                                 house_listing_count,
                                                 house_price_median,
                                                 house_variable_correlation,
                                                 list_amenities)

router = Blueprint('dashboard', __name__)


@router.route('/')
@cache.cached()
def page():
  return render_template(f'pages/index.html', stats=statistics())


# ---- top 10 districts by listing count
@router.route('/api/dashboard/charts/listing_count')
@cache.cached()
def chart_listing_count():
  return [dataclasses.asdict(house) for house in house_listing_count()]


# ---- top 10 districts by price box plot
@router.route('/api/dashboard/charts/median_price')
@cache.cached()
def chart_median_price():
  return [dataclasses.asdict(house) for house in house_price_median()]


# ---- correlation per variable to price
@router.route('/api/dashboard/charts/correlations')
@cache.cached()
def chart_correlations():
  return [dataclasses.asdict(house) for house in house_variable_correlation()]


# ---- amenities
@router.route('/api/dashboard/map/amenities')
@cache.cached()
def map_amenities():
  return [dataclasses.asdict(house) for house in list_amenities()]
