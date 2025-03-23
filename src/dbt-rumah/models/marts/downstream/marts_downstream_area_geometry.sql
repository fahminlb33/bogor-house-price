SELECT
    -- surrogate key
    {{ dbt_utils.generate_surrogate_key(['subdistrict', 'district', 'city', 'province']) }} AS area_sk,
    
    -- attributes
    subdistrict,
    district,
    city,
    province,
    ST_Y(center) AS latitude,
    ST_X(center) AS longitude,
    geom
FROM
    {{ ref("stg_administrative_area") }}
