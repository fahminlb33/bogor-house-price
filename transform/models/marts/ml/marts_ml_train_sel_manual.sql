{{ config(materialized='external', format='parquet') }}

WITH base AS (
	SELECT
		-- target
		price,

		-- features
		carport,
		dapur,
		daya_listrik,
		facility_ac,
		facility_keamanan,
		facility_laundry,
		facility_masjid,
		house_mat_bata_hebel,
		house_mat_bata_merah,
		jumlah_lantai,
		kamar_mandi,
		kamar_mandi_pembantu,
		kamar_pembantu,
		kamar_tidur,
		lebar_jalan,
		luas_bangunan,
		luas_tanah,
		ruang_makan,
		ruang_tamu,
		tag_cash_bertahap,
		tag_komplek,
		tag_kpr,
		tag_perumahan,
		tahun_dibangun,
	FROM
		{{ ref('int_ml_feature_outlier_removal') }}
)

SELECT
	*
FROM
	base
