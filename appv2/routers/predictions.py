from flask import Blueprint, request, g, render_template

from services.price_predictor import predictor, AVAILABLE_FACILITIES, AVAILABLE_HOUSE_MATERIAL, AVAILABLE_TAGS
from services import format_price_long
from services.repository.tracking import track_prediction

router = Blueprint('predictions', __name__)


@router.route('/predictions')
def page():
  return render_template(
      "pages/predict_price.html",
      facilities=AVAILABLE_FACILITIES,
      house_materials=AVAILABLE_HOUSE_MATERIAL,
      tags=AVAILABLE_TAGS)


@router.route('/predictions/predict', methods=['POST'])
def predict():
  # get input features
  input_features = {
      "luas_tanah": request.form.get("luas_tanah", 0),
      "luas_bangunan": request.form.get("luas_bangunan", 0),
      "daya_listrik": request.form.get("daya_listrik", 0),
      "jumlah_lantai": request.form.get("jumlah_lantai", 0),
      "kamar_mandi": request.form.get("kamar_mandi", 0),
      "kamar_tidur": request.form.get("kamar_tidur", 0),
      "kamar_pembantu": request.form.get("kamar_pembantu", 0),
      "kamar_mandi_pembantu": request.form.get("kamar_mandi_pembantu", 0),
      "dapur": request.form.get("dapur", 0),
      "lebar_jalan": request.form.get("lebar_jalan", 0),
      "carport": request.form.get("carport", 0),
      "fasilitas": request.form.getlist("fasilitas"),
      "house_material": request.form.getlist("house_material"),
      "tags": request.form.getlist("tags"),
  }

  # construct features
  X_pred = predictor.construct_features(input_features)

  # predict
  y_pred = predictor.predict(X_pred)
  price = format_price_long(y_pred[0] * 1_000_000)

  # track prediction
  track_prediction(input_features, y_pred[0])

  return render_template("pages/predict_price_result.html", price=price)
