import dataclasses

from flask import Blueprint, request, render_template, abort

from utils.shared import cache
from services.repository.data_mart import list_districts, statistics, house_histogram_bar

router = Blueprint('statistics', __name__)


@router.route('/statistics')
@cache.cached(make_cache_key=lambda:
              f"{request.path}/{request.args.get('district', 'default')}")
def page():
  # get all districts
  districts = list_districts()

  # get current district
  sk_district = request.args.get('district')
  if sk_district is None:
    # get first district if not specified
    sk_district = districts[0].district_sk

  # get statistics
  district_stats = statistics(sk_district)
  district_name = next(
      filter(lambda d: d.district_sk == sk_district, districts)).district

  return render_template(
      "pages/statistics.html",
      district_sk=sk_district,
      districts=districts,
      stats=district_stats,
      district_name=district_name)


# ---- top 10 districts by listing count
@router.route('/api/statistics/charts/price_histogram')
@cache.cached(make_cache_key=lambda:
              f"{request.path}/{request.args.get('district_sk', 'default')}")
def chart_listing_price_hist():
  # get current district
  sk_district = request.args.get('district_sk')
  if sk_district is None:
    abort(400)

  return [
      dataclasses.asdict(house) for house in house_histogram_bar(sk_district)
  ]
