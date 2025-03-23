SELECT
    -- surrogate key
    {{ dbt_utils.generate_surrogate_key(['name']) }} AS agent_company_sk,
    
    -- attributes
    COALESCE(name, 'Independen') AS company_name
FROM
    {{ ref("stg_rumah123_agent_companies") }}
GROUP BY 
    name
