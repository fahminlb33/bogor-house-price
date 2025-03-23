SELECT
	id AS reference_id,
	specs.*
FROM
	{{ source('raw_rumah123', 'houses') }}
