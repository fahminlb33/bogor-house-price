WITH
house_materials AS (
	SELECT
		reference_id,
		list(replace(material[11:], '_', ' ')) AS house_material
	FROM (
		UNPIVOT
			{{ ref('int_rumah123_house_materials') }}
		ON
			COLUMNS(* EXCLUDE (reference_id))
		INTO
			NAME material
			VALUE material_count
	)
	GROUP BY
		reference_id, material_count
	HAVING
		material_count > 0
),
house_floor_materials AS (
	SELECT
		reference_id,
		list(material[11:]) AS floor_material
	FROM (
		UNPIVOT
			{{ ref('int_rumah123_floor_materials') }}
		ON
			COLUMNS(* EXCLUDE (reference_id))
		INTO
			NAME material
			VALUE material_count
	)
	GROUP BY
		reference_id, material_count
	HAVING
		material_count > 0
),
house_tags AS (
	SELECT
		reference_id,
		list(replace(tag[5:], '_', ' ')) AS tags
	FROM (
		UNPIVOT
			{{ ref('int_rumah123_tags') }}
		ON
			COLUMNS(* EXCLUDE (reference_id))
		INTO
			NAME tag
			VALUE tag_count
	)
	GROUP BY
		reference_id, tag_count
	HAVING
		tag_count > 0
),
house_facility AS (
	SELECT
		reference_id,
		list(replace(facility[10:], '_', ' ')) AS facilities
	FROM (
		UNPIVOT
			{{ ref('int_rumah123_facilities') }}
		ON
			COLUMNS(* EXCLUDE (reference_id))
		INTO
			NAME facility
			VALUE facility_count
	)
	GROUP BY
		reference_id, facility_count
	HAVING
		facility_count > 0
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
staging AS (
	SELECT
		*
	FROM
		{{ ref('stg_rumah123_houses') }} houses
	LEFT JOIN
		{{ ref('int_rumah123_specs') }} house_specs ON house_specs.reference_id = houses.id
	LEFT JOIN
		house_materials ON house_materials.reference_id = houses.id
	LEFT JOIN
		house_floor_materials ON house_floor_materials.reference_id = houses.id
	LEFT JOIN
		house_tags ON house_tags.reference_id = houses.id
	LEFT JOIN
		house_facility ON house_facility.reference_id = houses.id
	LEFT JOIN
		house_images ON house_images.reference_id = houses.id
),
final AS (
	SELECT
		DISTINCT ON (id) *
	FROM
		staging
)

SELECT
	* EXCLUDE(last_modified, scraped_at, reference_id, reference_id_1, reference_id_2, reference_id_3, reference_id_4, reference_id_5)
FROM
	final
WHERE
	price IS NOT NULL
