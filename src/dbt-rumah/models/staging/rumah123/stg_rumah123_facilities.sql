WITH
house_facilities AS (
	SELECT
        id as reference_id,
        unnest(facilities) AS facility_name
    FROM
        {{ source('raw_rumah123', 'houses') }}
),
cleaned AS (
	SELECT
		reference_id,
		clean_facility(facility_name) AS facility
	FROM
		house_facilities
)

SELECT
	*
FROM
	cleaned
WHERE
	facility IS NOT NULL
