SELECT
	specs.*,
	id AS reference_id
FROM
	{{ source('raw_rumah123', 'houses') }}
