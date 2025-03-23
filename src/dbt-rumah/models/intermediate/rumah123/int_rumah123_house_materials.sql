WITH
house_materials AS (
	SELECT
        reference_id,
        unnest(string_split(lower(material_bangunan), ',')) AS material
    FROM
        {{ ref('int_rumah123_specs') }}
)

SELECT
	reference_id,
    replace(trim(material), ' ', '_') AS house_material
FROM
    house_materials
