WITH
amenities_unnested AS (
	SELECT
        rel,
        amenity,
        unnest(data.elements) AS element
    FROM
        {{ source('raw_osm', 'raw_osm_amenities') }}
)

SELECT
    element.id,
	rel,
	amenity,
	element.type AS object_type,
	coalesce(element.lat, element.center.lat) AS lat,
	coalesce(element.lon, element.center.lon) AS lon,
	element.nodes,
	element.tags
FROM
    amenities_unnested
