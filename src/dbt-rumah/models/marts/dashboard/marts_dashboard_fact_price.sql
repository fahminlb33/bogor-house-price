WITH
final AS (
    SELECT
		-- key
        {{ dbt_utils.generate_surrogate_key(['district', 'city']) }} AS district_sk,

        -- attributes
        price * 1000000 AS price,
		installment
	FROM
		{{ ref("stg_rumah123_houses") }}
	WHERE
		price IS NOT NULL
)

SELECT * FROM final
