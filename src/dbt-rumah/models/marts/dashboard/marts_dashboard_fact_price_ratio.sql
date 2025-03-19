WITH
staging AS (
    SELECT
        *
	FROM
		{{ ref("int_rumah123_houses") }} houses
	LEFT JOIN
		{{ ref("int_rumah123_specs") }} specs ON specs.reference_id = houses.id
),
final AS (
	SELECT
		-- key
        {{ dbt_utils.generate_surrogate_key(['subdistrict', 'city']) }} AS district_sk,

		-- attributes
		(price * 1000000) / luas_tanah AS price_per_land_area,
		(price * 1000000) / luas_bangunan AS price_per_building_area
	FROM
		staging
)

SELECT * FROM final
