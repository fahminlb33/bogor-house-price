WITH
house_materials AS (
	SELECT
        unnest(string_split(lower(material_bangunan), ',')) AS material,
        reference_id
    FROM
        {{ ref('int_rumah123_norm_specs') }}
)

SELECT
    replace(trim(material), ' ', '_') AS house_material,
	reference_id
FROM
    house_materials
