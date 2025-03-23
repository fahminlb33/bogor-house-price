SELECT
    -- surrogate key
    {{ dbt_utils.generate_surrogate_key(['tag']) }} AS tag_sk,
    
    -- attributes
    title_case(replace(tag, '_', ' ')) AS tag_name,
FROM
    {{ ref("stg_rumah123_tags") }}
GROUP BY 
    tag
