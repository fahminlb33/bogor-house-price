{%- set cols = dbt_utils.get_column_values(ref('int_osm_amenities'), 'category') -%}

WITH
staging AS (
    SELECT
        admin_area.subdistrict,
        {{ dbt_utils.pivot(
            'amenities.category',
            cols,
            agg='sum'
        ) }}
    FROM
        {{ ref('int_osm_amenities') }} amenities
    LEFT JOIN 
        {{ ref('stg_administrative_area')}} admin_area ON ST_Contains(admin_area.geom, amenities.center)
    GROUP BY
        admin_area.subdistrict
)

SELECT 
    subdistrict, 
    struct_pack(
        {% for column in cols %}
			{{ "{} := {}".format(column, column) }}{{ "," if not loop.last }}
		{% endfor %}
    ) AS amenities
FROM
    staging
