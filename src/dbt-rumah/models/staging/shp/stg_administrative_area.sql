SELECT
    NAMOBJ as subdistrict, 
    WADMKC as district, 
    WADMKK as city,
    WADMPR as province,
    ST_Centroid(geom) as center,
    geom
FROM 
    {{ source('raw_shp', 'kota_bogor') }}
UNION ALL
SELECT
    NAMOBJ as subdistrict, 
    WADMKC as district, 
    WADMKK as city,
    WADMPR as province,
    ST_Centroid(geom) as center,
    geom
FROM 
    {{ source('raw_shp', 'kab_bogor') }}
