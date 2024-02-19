import streamlit as st

st.set_page_config(
    page_title="Prediksi Harga Rumah - NyariRumah",
    page_icon="👋",
)

st.title("Hello world")

animal_shelter = ['cat', 'dog', 'rabbit', 'bird']

with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        animal1 = st.number_input('Luas tanah')
        animal2 = st.selectbox('Animal', animal_shelter)

    with col2:
        animal2 = st.number_input('Luas bangunan')

    st.form_submit_button("Prediksi")
