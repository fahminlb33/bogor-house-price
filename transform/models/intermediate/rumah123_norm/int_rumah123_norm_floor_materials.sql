WITH
house_materials AS (
	SELECT
        unnest(string_split(lower(material_lantai), ',')) AS material,
        reference_id
    FROM
        {{ ref('int_rumah123_specs_norm') }}
)

SELECT
    replace(trim(material), ' ', '_') AS floor_material,
	reference_id
FROM
    house_materials
