WITH
house_facilities AS (
	SELECT
        unnest(facilities) AS facility_name,
        id as reference_id
    FROM
        {{ source('raw_rumah123', 'houses') }}
),
cleaned AS (
	SELECT
		clean_facility(facility_name) AS facility,
		reference_id
	FROM
		house_facilities
)

SELECT
	*
FROM
	cleaned
WHERE
	facility IS NOT NULL

