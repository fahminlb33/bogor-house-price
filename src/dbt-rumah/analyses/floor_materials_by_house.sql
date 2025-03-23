SELECT
	reference_id,
	{{ dbt_utils.pivot(
		'floor_material',
		dbt_utils.get_column_values(ref('int_rumah123_floor_materials'), 'floor_material'),
		agg='max'
	) }}
FROM
	{{ ref('int_rumah123_floor_materials') }}
GROUP BY
	reference_id
