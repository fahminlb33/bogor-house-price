WITH
house_agent AS (
	SELECT
        agent.*,
        id as reference_id
    FROM
        {{ source('raw_rumah123', 'houses') }}
)

SELECT
    clean_agency_company(company.name) AS name,
	company.url as url,
	reference_id
FROM
    house_agent
