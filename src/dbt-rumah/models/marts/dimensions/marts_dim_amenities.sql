SELECT
    -- surrogate key
    {{ dbt_utils.generate_surrogate_key(['amenity']) }} AS amenity_sk,
    
    -- attributes
    title_case(replace(category, '_', ' ')) AS category_name,
    title_case(replace(amenity, '_', ' ')) AS amenity_name
FROM
    {{ ref("int_osm_amenities") }}
GROUP BY 
    category, amenity
