WITH
house_full AS (
	SELECT
		*
	FROM
		{{ ref('stg_rumah123_houses') }} AS houses
	LEFT JOIN
		{{ ref('int_rumah123_specs') }}  AS house_specs ON house_specs.reference_id = houses.id
	LEFT JOIN
		{{ ref('int_rumah123_house_materials') }} AS  house_materials ON house_materials.reference_id = houses.id
	LEFT JOIN
		{{ ref('int_rumah123_floor_materials') }}  AS floor_materials ON floor_materials.reference_id = houses.id
	LEFT JOIN
		{{ ref('int_rumah123_tags') }}  AS house_tags ON house_tags.reference_id = houses.id
	LEFT JOIN
		{{ ref('int_rumah123_facilities') }} house_facilities ON house_facilities.reference_id = houses.id
)

SELECT
	* EXCLUDE(reference_id, reference_id_1, reference_id_2, reference_id_3, reference_id_4)
FROM
	house_full
