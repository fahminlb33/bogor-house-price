SELECT
	reference_id,
	{{ dbt_utils.pivot(
		'tag',
		dbt_utils.get_column_values(ref('stg_rumah123_tags'), 'tag'),
		prefix='tag_'
	) }}
FROM
	{{ ref('stg_rumah123_tags') }}
GROUP BY
	reference_id
