WITH
house_tags AS (
	SELECT
        id as reference_id,
        unnest(tags) AS tag_name
    FROM
        {{ source('raw_rumah123', 'houses') }}
)

SELECT
	reference_id,
    unnest(string_split(replace(lower(tag_name), ' ', '_'), '/')) AS tag
FROM
    house_tags
