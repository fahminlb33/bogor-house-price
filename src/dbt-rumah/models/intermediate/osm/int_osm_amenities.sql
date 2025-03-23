SELECT
    id, 
    element, 
    amenity, 
    name, 
    ST_Centroid(geom) as center,
    geom,
    CASE 
        WHEN amenity IN ('bar', 'biergarten', 'cafe', 'fast_food', 'food_court', 'ice_cream', 'pub', 'restaurant') THEN 'sustenance'
        WHEN amenity IN ('college', 'dancing_school', 'driving_school', 'first_aid_school', 'kindergarten', 'language_school', 'library', 'surf_school', 'toy_library', 'research_institute', 'training', 'music_school', 'school', 'traffic_park', 'university', 'prep_school') THEN 'education'
        WHEN amenity IN ('bicycle_parking', 'bicycle_repair_station', 'bicycle_rental', 'bicycle_wash', 'boat_rental', 'boat_sharing', 'bus_station', 'car_rental', 'car_sharing', 'car_wash', 'compressed_air', 'vehicle_inspection', 'charging_station', 'driver_training', 'ferry_terminal', 'fuel', 'grit_bin', 'motorcycle_parking', 'parking', 'parking_entrance', 'parking_space', 'taxi', 'weighbridge', 'motorcycle_taxi') THEN 'transportation'
        WHEN amenity IN ('atm', 'payment_terminal', 'bank', 'bureau_de_change') THEN 'financial'
        WHEN amenity IN ('baby_hatch', 'clinic', 'dentist', 'doctors', 'hospital', 'nursing_home', 'pharmacy', 'social_facility', 'veterinary', 'animal_boarding', 'animal_shelter', 'animal_breeding', 'childcare', 'public_bath') THEN 'healthcare'
        WHEN amenity IN ('arts_centre', 'brothel', 'casino', 'cinema', 'community_centre', 'conference_centre', 'events_venue', 'exhibition_centre', 'fountain', 'gambling', 'love_hotel', 'music_venue', 'nightclub', 'planetarium', 'public_bookcase', 'social_centre', 'stripclub', 'studio', 'swingerclub', 'theatre', 'hunting_stand', 'karaoke_box', 'karaoke') THEN 'entertainment'
        WHEN amenity IN ('courthouse', 'fire_station', 'police', 'post_box', 'post_depot', 'post_office', 'prison', 'ranger_station', 'townhall', 'public_building', 'grave_yard') THEN 'public_service'
        WHEN amenity IN ('place_of_worship', 'bbq', 'bench', 'dog_toilet', 'dressing_room', 'drinking_water', 'give_box', 'mailroom', 'parcel_locker', 'shelter', 'shower', 'telephone', 'toilets', 'water_point', 'watering_place', 'letter_box', 'security_booth') THEN 'facilities'
        WHEN amenity IN ('sanitary_dump_station', 'recycling', 'waste_basket', 'waste_disposal', 'waste_transfer_station') THEN 'waste_management'
        WHEN amenity IN ('commercial', 'retail', 'marketplace', 'beauty', 'internet_cafe', 'industrial', 'coworking_space', 'office') THEN 'commercial'
        ELSE 'other'
    END AS category
FROM 
    {{ ref('stg_osm_amenities') }}
