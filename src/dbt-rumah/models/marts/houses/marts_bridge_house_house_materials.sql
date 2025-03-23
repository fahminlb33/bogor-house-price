SELECT
    -- key
    reference_id AS house_id,
    {{ dbt_utils.generate_surrogate_key(['house_material']) }} AS house_material_sk
FROM 
    {{ ref('int_rumah123_house_materials') }}
