WITH
house_agent AS (
	SELECT
        id as reference_id,
        agent.*
    FROM
        {{ source('raw_rumah123', 'houses') }}
)

SELECT
	reference_id,
    clean_agency_company(company.name) AS name,
	company.url as url
FROM
    house_agent
