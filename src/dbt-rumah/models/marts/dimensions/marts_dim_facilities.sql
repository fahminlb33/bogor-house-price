SELECT
    -- surrogate key
    {{ dbt_utils.generate_surrogate_key(['facility']) }} AS facility_sk,
    
    -- attributes
    title_case(replace(facility, '_', ' ')) AS facility_name,
FROM
    {{ ref("stg_rumah123_facilities") }}
GROUP BY
    facility
