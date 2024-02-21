{{ config(materialized='external', format='parquet') }}

WITH
houses AS (
	SELECT
		-- metadata
		id,
		district,
		city,
		description,
		url,

		-- features
		price,
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
),
house_images AS (
	SELECT
		reference_id,
		first(photo_url) AS main_image_url
	FROM
		{{ ref('stg_rumah123_images') }}
	GROUP BY
		reference_id
),
final AS (
	SELECT
		*
	FROM
		houses
	LEFT JOIN
		house_images
	ON
		houses.id = house_images.reference_id
)

SELECT
	* EXCLUDE (reference_id)
FROM
	final
LIMIT 10
