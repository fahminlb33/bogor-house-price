WITH
amenities_unnested AS (
	SELECT
        rel,
        amenity,
        unnest(data.elements) AS element
    FROM
        {{ source('raw_osm', 'raw_osm_amenities') }}
),
amenities_clear AS (
	SELECT
		element.id,
		rel,
		amenity,
		element.type AS object_type,
		coalesce(element.lat, element.center.lat) AS lat,
		coalesce(element.lon, element.center.lon) AS lon,
		element.nodes,
		element.tags,

		CASE
			WHEN amenity IN (
				'bar',
				'biergarten',
				'cafe',
				'fast_food',
				'food_court',
				'ice_cream',
				'pub',
				'restaurant'
			) THEN 'sustenance'
			WHEN amenity IN (
				'college',
				'dancing_school',
				'driving_school',
				'first_aid_school',
				'kindergarten',
				'language_school',
				'library',
				'surf_school',
				'toy_library',
				'research_institute',
				'training',
				'music_school',
				'school',
				'traffic_park',
				'university'
			) THEN 'education'
			WHEN amenity IN (
				'bicycle_parking',
				'bicycle_repair_station',
				'bicycle_rental',
				'bicycle_wash',
				'boat_rental',
				'boat_sharing',
				'bus_station',
				'car_rental',
				'car_sharing',
				'car_wash',
				'compressed_air',
				'vehicle_inspection',
				'charging_station',
				'driver_training',
				'ferry_terminal',
				'fuel',
				'grit_bin',
				'motorcycle_parking',
				'parking',
				'parking_entrance',
				'parking_space',
				'taxi',
				'weighbridge'
			) THEN 'transportation'
			WHEN amenity IN (
				'atm',
				'payment_terminal',
				'bank',
				'bureau_de_change'
			) THEN 'financial'
			WHEN amenity IN (
				'baby_hatch',
				'clinic',
				'dentist',
				'doctors',
				'hospital',
        		'nursing_home',
				'pharmacy',
				'social_facility',
				'veterinary'
			) THEN 'healthcare'
			WHEN amenity IN (
				'arts_centre',
				'brothel',
				'casino',
				'cinema',
				'community_centre',
        		'conference_centre',
				'events_venue',
				'exhibition_centre',
				'fountain',
        		'gambling',
				'love_hotel',
				'music_venue',
				'nightclub',
				'planetarium',
        		'public_bookcase',
				'social_centre',
				'stripclub',
				'studio',
        		'swingerclub',
				'theatre'
			) THEN 'entertainment'
			WHEN amenity IN (
				'courthouse',
				'fire_station',
				'police',
				'post_box',
				'post_depot',
        		'post_office',
				'prison',
				'ranger_station',
				'townhall'
			) THEN 'public_service'
			WHEN amenity IN (
				'bbq',
				'bench',
				'dog_toilet',
				'dressing_room',
				'drinking_water',
        		'give_box',
				'mailroom',
				'parcel_locker',
				'shelter',
				'shower',
        		'telephone',
				'toilets',
				'water_point',
				'watering_place'
			) THEN 'facilities'
			WHEN amenity IN (
				'sanitary_dump_station',
				'recycling',
				'waste_basket',
				'waste_disposal',
        		'waste_transfer_station'
			) THEN 'waste_management'
			ELSE 'other'
		END AS category
	FROM
		amenities_unnested
)

SELECT
	*
FROM
	amenities_clear
