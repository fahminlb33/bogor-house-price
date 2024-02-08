from flask import (Blueprint, flash, g, redirect, render_template, request,
                   session, url_for)

from infrastructure import constants

router = Blueprint('predictions', __name__, url_prefix='/predictions')


@router.route("/")
def index():
    params = {
        "FLOOR_MATERIAL_ITEMS": constants.FLOOR_MATERIAL_ITEMS,
        "HOUSE_MATERIAL_ITEMS": constants.HOUSE_MATERIAL_ITEMS,
        "FACILITY_ITEMS": constants.FACILITY_ITEMS,
        "TAG_ITEMS": constants.TAG_ITEMS,
        "CONDITION_ITEMS": constants.CONDITION_ITEMS,
        "WATER_SOURCE_ITEMS": constants.WATER_SOURCE_ITEMS,
        "CONCEPT_ITEMS": constants.CONCEPT_ITEMS,
        "VIEW_ITEMS": constants.VIEW_ITEMS,
        "CERTIFICATE_ITEMS": constants.CERTIFICATE_ITEMS,
    }

    return render_template("prediction.html", **params)


@router.route("/predict", methods=["POST"])
def predict():
    print(request.form)
    params = {
        "predicted_price": "100 juta",
        "predicted_price_range": "50-150 juta",
        "predicted_error": "10 juta",
    }

    return render_template("prediction_result.html", **params)
