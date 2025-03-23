SELECT
    -- surrogate key
    {{ dbt_utils.generate_surrogate_key(['amenity']) }} AS amenity_sk,
    
    -- attributes
    title_case(replace(amenity, '_', ' ')) AS amenity_name
FROM
    {{ ref("stg_osm_amenities") }}
GROUP BY 
    amenity
