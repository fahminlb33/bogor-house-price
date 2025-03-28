WITH
house_agent AS (
	SELECT
        id as reference_id,
        agent.*
    FROM
        {{ source('raw_rumah123', 'houses') }}
)

SELECT
    mask_name(name) AS name,
	mask_phone(phone) AS phone,
	sha256(concat(name, phone)) AS agent_hash,
	reference_id
FROM
    house_agent
