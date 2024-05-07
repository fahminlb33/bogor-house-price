{{ config(materialized='external', format='csv') }}

SELECT
    unnest(images) AS photo_url,
	id AS reference_id
FROM
    {{ source('raw_rumah123', 'houses') }}
