SELECT
    element, 
    id, 
    amenity, 
    name, 
    ST_Centroid(geom) as center,
    geom
FROM 
    {{ source('raw_osm', 'amenities') }}
