SELECT
    -- surrogate key
    {{ dbt_utils.generate_surrogate_key(['floor_material']) }} AS floor_material_sk,
    
    -- attributes
    title_case(replace(floor_material, '_', ' ')) AS floor_material,
FROM
    {{ ref("int_rumah123_floor_materials") }}
GROUP BY
    floor_material
