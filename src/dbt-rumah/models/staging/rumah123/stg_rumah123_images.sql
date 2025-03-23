SELECT
	id AS reference_id,
    unnest(images) AS image_url
FROM
    {{ source('raw_rumah123', 'houses') }}
