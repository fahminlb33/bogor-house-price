SELECT
    -- key
    reference_id AS house_id,
    {{ dbt_utils.generate_surrogate_key(['floor_material']) }} AS floor_material_sk
FROM 
    {{ ref('int_rumah123_floor_materials') }}
