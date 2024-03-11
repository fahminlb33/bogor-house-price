SELECT
	reference_id,
	{{ dbt_utils.pivot(
		'house_material',
		dbt_utils.get_column_values(ref('int_rumah123_norm_house_materials'), 'house_material'),
		prefix='house_mat_',
		agg='max'
	) }}
FROM
	{{ ref('int_rumah123_norm_house_materials') }}
GROUP BY
	reference_id
