WITH
final AS (
	SELECT
        rel,
        amenity,
        unnest(data.elements) AS element
    FROM
        {{ source('raw_osm', 'amenities') }}
)
SELECT
	*
FROM
	final
