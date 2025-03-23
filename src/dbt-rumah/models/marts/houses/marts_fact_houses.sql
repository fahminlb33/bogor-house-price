WITH
images AS (
    SELECT
        DISTINCT reference_id,
        image_url
    FROM
        {{ ref('stg_rumah123_images') }}
)

SELECT 
    -- key
    house.id AS house_id,
    {{ dbt_utils.generate_surrogate_key(['admin_area.subdistrict', 'admin_area.district', 'admin_area.city', 'admin_area.province']) }} AS area_sk,
    {{ dbt_utils.generate_surrogate_key(['agent_company.name']) }} AS agent_company_sk,

    -- attributes
    house.price * 1000000 AS price,
    house.installment * 1000000 AS installment,

    house_specs.kamar_tidur,
    house_specs.kamar_mandi,
    house_specs.kamar_pembantu,
    house_specs.kamar_mandi_pembantu,
    house_specs.dapur,
    house_specs.luas_tanah,
	house_specs.luas_bangunan,

    house_specs.jumlah_lantai,
    house_specs.garasi,
    house_specs.carport,
    house_specs.lebar_jalan,
    house_specs.tahun_dibangun,
    house_specs.tahun_di_renovasi,
    house_specs.daya_listrik,

    house_specs.hadap,
    house_specs.sertifikat,
    house_specs.sumber_air,
    house_specs.pemandangan,
    house_specs.tipe_properti,
    house_specs.konsep_dan_gaya_rumah,
    house_specs.kondisi_properti,
    house_specs.kondisi_perabotan,

    house_specs.hook,
    house_specs.ruang_tamu,
    house_specs.ruang_makan,
    house_specs.terjangkau_internet,

    house.description,
    house.last_modified,
    house.url,
    images.image_url
FROM 
    {{ ref('int_rumah123_houses') }} house
INNER JOIN
    {{ ref('int_rumah123_specs') }} house_specs ON house.id = house_specs.reference_id
LEFT JOIN
    images ON house.id = images.reference_id
LEFT JOIN 
    -- subdistrict is not unique!
    -- some mismatch is expected
    {{ ref('stg_administrative_area')}} admin_area ON house.subdistrict = admin_area.subdistrict
LEFT JOIN
    {{ ref('stg_rumah123_agent_companies') }} agent_company ON house.id = agent_company.reference_id
