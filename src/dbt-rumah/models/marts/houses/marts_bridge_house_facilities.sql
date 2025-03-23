SELECT
    -- key
    reference_id AS house_id,
    {{ dbt_utils.generate_surrogate_key(['facility']) }} AS facility_sk
FROM 
    {{ ref('stg_rumah123_facilities') }}
