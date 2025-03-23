SELECT
    -- surrogate key
    {{ dbt_utils.generate_surrogate_key(['house_material']) }} AS house_material_sk,
    
    -- attributes
    title_case(replace(house_material, '_', ' ')) AS house_material,
FROM
    {{ ref("int_rumah123_house_materials") }}
GROUP BY
    house_material
