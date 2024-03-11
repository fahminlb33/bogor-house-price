{{ config(materialized='external', format='csv') }}

WITH
final AS (
    SELECT
		id,
		rel,
		amenity,
		object_type,
		category,
		lat,
		lon
    FROM
        {{ ref('int_spatial_amenities') }}
)

SELECT * FROM final
