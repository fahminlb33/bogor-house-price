WITH base AS (
	SELECT
		reference_id,
		{{ dbt_utils.pivot(
			'facility',
			dbt_utils.get_column_values(ref('stg_rumah123_facilities'), 'facility'),
			prefix='facility_',
			agg='max'
		) }}
	FROM
		{{ ref('stg_rumah123_facilities') }}
	GROUP BY
		reference_id
)

SELECT
	*
FROM
	base
