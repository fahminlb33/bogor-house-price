SELECT
    -- key
    {{ dbt_utils.generate_surrogate_key(['amenities.amenity']) }} AS amenity_sk,
    {{ dbt_utils.generate_surrogate_key(['admin_area.subdistrict', 'admin_area.district', 'admin_area.city', 'admin_area.province']) }} AS area_sk,

    -- attributes
    amenities.name,
    ST_Y(amenities.center) AS latitude,
    ST_X(amenities.center) AS longitude
FROM 
    {{ ref('int_osm_amenities') }} amenities
LEFT JOIN 
    {{ ref('stg_administrative_area')}} admin_area ON ST_Contains(admin_area.geom, amenities.center)
