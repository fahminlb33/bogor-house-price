SELECT
	reference_id,
	{{ dbt_utils.pivot(
		'house_material',
		dbt_utils.get_column_values(ref('int_rumah123_house_materials'), 'house_material'),
		agg='max'
	) }}
FROM
	{{ ref('int_rumah123_house_materials') }}
GROUP BY
	reference_id
