SELECT
	reference_id,
	{{ dbt_utils.pivot(
		'floor_material',
		dbt_utils.get_column_values(ref('int_rumah123_norm_floor_materials'), 'floor_material'),
		prefix='floor_mat_',
		agg='max'
	) }}
FROM
	{{ ref('int_rumah123_norm_floor_materials') }}
GROUP BY
	reference_id
