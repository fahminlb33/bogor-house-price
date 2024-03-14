{{ config(materialized='external', format='parquet') }}

WITH base AS (
	SELECT
		-- target
		price,

		-- features
		luas_tanah,
		luas_bangunan,
		kamar_tidur,
		kamar_mandi,
		kamar_pembantu,
		kamar_mandi_pembantu,

		daya_listrik,
		jumlah_lantai,
		lebar_jalan,
		carport,
		dapur,
		ruang_makan,
		ruang_tamu,

		facility_ac,
		facility_keamanan,
		facility_laundry,
		facility_masjid,

		house_mat_bata_hebel,
		house_mat_bata_merah,

		tag_cash_bertahap,
		tag_komplek,
		tag_kpr,
		tag_perumahan,
	FROM
		{{ ref('int_ml_feature_outlier_removal') }}
)

SELECT
	*
FROM
	base
