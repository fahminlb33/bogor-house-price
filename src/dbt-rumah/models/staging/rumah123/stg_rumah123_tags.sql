WITH
house_tags AS (
	SELECT
        unnest(tags) AS tag_name,
        id as reference_id
    FROM
        {{ source('raw_rumah123', 'houses') }}
)

SELECT
    unnest(string_split(replace(lower(tag_name), ' ', '_'), '/')) AS tag,
	reference_id
FROM
    house_tags
