SELECT
    source,
    target
FROM 
    {{ source('raw_osm', 'district') }}
