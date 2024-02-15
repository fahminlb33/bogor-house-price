WITH
amenities_unnested AS (
	SELECT
        rel,
        amenity,
        arrayJoin(data.elements) AS elem
    FROM
        {{ source('raw_osm', 'raw_osm_amenities') }}
),
amenities_coords AS (
	SELECT
        rel,
        amenity,
        elem.1 AS center,
        elem.2 as id,
        elem.3 AS lat,
        elem.4 as lon,
        elem.5 as nodes,
        elem.6 as tags,
        elem.7 as object_type
    FROM
        amenities_unnested
)

SELECT
    id,
    rel,
    amenity,
    object_type,
    COALESCE(lat, center.1) AS lat,
    COALESCE(lon, center.2) AS lon,
    nodes,
    tags
FROM
    amenities_coords
