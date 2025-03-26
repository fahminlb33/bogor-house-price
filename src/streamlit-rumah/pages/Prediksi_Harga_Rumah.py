import streamlit as st

from utils.helpers import format_price_long
from utils.price_predictor import (
    get_subdistricts,
    predict_price,
)


def main():
    #
    # Page configuration
    #

    # set page config
    st.set_page_config(
        page_title="Prediksi Harga Rumah - NyariRumah",
        page_icon="👋",
    )

    st.sidebar.markdown(
        """
        Prediksi dilakukan menggunakan model LightGBM.
        
        Source code: [klik disini.](https://github.com/fahminlb33/bogor-house-price/blob/master/notebooks/train-eval.ipynb)
        """
    )

    # load custom styles
    st.markdown(
        """
        <style>
        .text-center {
            text-align: center;
        }

        .mt-atas {
            margin-top: 40px;
        }

        .text-prediction {
            font-size: 5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    #
    # Page contents
    #

    st.title("🏡Prediksi Harga Rumah")
    st.markdown("""
                Selamat datang ke layanan prediksi harga rumah! Masukkan informasi spesifikasi rumah yang ingin Anda beli pada form berikut.
                """)

    # input features

    input_features = {}
    with st.form("predict_form", border=False):
        st.subheader("Spesifikasi Rumah")

        col1, col2 = st.columns(2)
        with col1:
            input_features["luas_tanah"] = st.number_input("Luas tanah", step=1)
            input_features["kamar_tidur"] = st.number_input("Kamar tidur", step=1)
            input_features["daya_listrik"] = st.number_input("Daya listrik", step=1)
            input_features["jumlah_lantai"] = st.number_input("Jumlah lantai", step=1)

        with col2:
            input_features["luas_bangunan"] = st.number_input("Luas bangunan", step=1)
            input_features["kamar_mandi"] = st.number_input("Kamar mandi", step=1)
            input_features["tahun_dibangun"] = st.number_input("Tahun Dibangun", step=1)
            input_features["subdistrict"] = st.selectbox(
                "Kelurahan", get_subdistricts()
            )

        st.text("")
        submit_res = st.form_submit_button(
            "Prediksi", type="primary", use_container_width=True
        )

    #
    # Prediction output
    #

    if submit_res:
        # construct features
        X_pred = {
            "subdistrict": [input_features["subdistrict"]],
            "luas_tanah": [input_features["luas_tanah"]],
            "luas_bangunan": [input_features["luas_bangunan"]],
            "jumlah_lantai": [input_features["jumlah_lantai"]],
            "tahun_dibangun": [input_features["tahun_dibangun"]],
            "daya_listrik": [input_features["daya_listrik"]],
            "land_building_ratio": [
                input_features["luas_tanah"] / max(input_features["luas_bangunan"], 1)
            ],
            "total_bedrooms": [input_features["kamar_tidur"]],
            "total_bathrooms": [input_features["kamar_mandi"]],
            "building_area_floor_ratio": [
                input_features["luas_bangunan"]
                / max(input_features["jumlah_lantai"], 1)
            ],
        }

        # predict
        price = format_price_long(predict_price(X_pred))

        # show prediction
        st.markdown(
            '<div class="text-center mt-atas">Hasil Prediksi', unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="text-center text-prediction">{price}</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    # bootstrap
    main()
