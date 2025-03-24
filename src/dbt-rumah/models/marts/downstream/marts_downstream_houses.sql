WITH
house_materials AS (
	SELECT
		reference_id,
		ARRAY_AGG(house_material) AS house_materials,
	FROM 
		{{ ref('int_rumah123_house_materials') }}
	GROUP BY
		reference_id
),
floor_materials AS (
	SELECT
		reference_id,
		ARRAY_AGG(floor_material) AS floor_materials,
	FROM 
		{{ ref('int_rumah123_floor_materials') }}
	GROUP BY
		reference_id
),
tags AS (
	SELECT
		reference_id,
		ARRAY_AGG(tag) AS tags
	FROM
		{{ ref('stg_rumah123_tags') }}
	GROUP BY
		reference_id
),
facilities AS (
	SELECT
		reference_id,
		ARRAY_AGG(facility) AS facilities
	FROM
		{{ ref('stg_rumah123_facilities') }}
	GROUP BY
		reference_id
),
images AS (
	SELECT
		reference_id,
		ARRAY_AGG(image_url) AS image_urls
	FROM
		{{ ref('stg_rumah123_images') }}
	GROUP BY
		reference_id
),
staging AS (
	SELECT
		houses.id,
    	houses.price * 1000000 AS price,
    	houses.installment * 1000000 AS installment,
		houses.subdistrict,
		houses.city,
		houses.description,
		houses.url,

		specs.kamar_tidur,
		specs.kamar_mandi,
		specs.kamar_pembantu,
		specs.kamar_mandi_pembantu,
		specs.dapur,
		specs.luas_tanah,
		specs.luas_bangunan,

		specs.jumlah_lantai,
		specs.garasi,
		specs.carport,
		specs.lebar_jalan,
		specs.tahun_dibangun,
		specs.tahun_di_renovasi,
		specs.daya_listrik,

		specs.hadap,
		specs.sertifikat,
		specs.sumber_air,
		specs.pemandangan,
		specs.tipe_properti,
		specs.konsep_dan_gaya_rumah,
		specs.kondisi_properti,
		specs.kondisi_perabotan,

		specs.hook,
		specs.ruang_tamu,
		specs.ruang_makan,
		specs.terjangkau_internet,

		tags.tags,
		facilities.facilities,
		house_materials.house_materials,
		floor_materials.floor_materials,
		images.image_urls,
		amenities.amenities
	FROM
		{{ ref('int_rumah123_houses') }} houses
	LEFT JOIN
		{{ ref('int_rumah123_specs') }} specs ON specs.reference_id = houses.id
	LEFT JOIN
		tags ON tags.reference_id = houses.id
	LEFT JOIN
		facilities ON facilities.reference_id = houses.id
	LEFT JOIN
		house_materials ON house_materials.reference_id = houses.id
	LEFT JOIN
		floor_materials ON floor_materials.reference_id = houses.id
	LEFT JOIN
		images ON images.reference_id = houses.id
	LEFT JOIN 
		-- subdistrict is not unique!
		-- some mismatch is expected
		{{ ref('int_osm_amenities_area_counts')}} amenities ON houses.subdistrict = amenities.subdistrict
)

SELECT
	DISTINCT ON (id) *
FROM
	staging
WHERE
	price IS NOT NULL
