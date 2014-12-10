-- File: pickups-aggregated.sql
-- ---------------------------------
-- SQL query to create a table with taxi
-- pickup counts aggregated by time and location
-- windows.
-- Time is divided into 1 hour intervals.
-- Location is divided into squares 0.01 degrees lat/long.
-- There are 200 location zones per row, and a total of
-- 100 rows (20,000 location zones in total).

-- We expect there to be at most
-- 20,000 zones * 24 hrs/day * 31 days = 14,880,000 rows.
-- In reality, there are 143,649 rows, which is reasonable
-- because not all zones are actually in NYC, and not all
-- NYC zones have at least one pickup every hour.

-- Runs in < 75 seconds on the first trip_data table.

USE taxi_pickups;


-- #### Create a temporary pickups_aggregated_manhattan_temp table
-- that does not include rows with zero pickups. (This whole
-- step takes ~6 minutes.)
DROP TABLE IF EXISTS pickups_aggregated_manhattan_temp;

CREATE TABLE pickups_aggregated_manhattan_temp (
    -- We define a fixed time interval (1 hour), so no need for end time.
    start_datetime DATETIME NOT NULL,

    -- Zone IDs run west to east (-75 to -73 degrees),
    -- and south to north (40 to 41 degrees).
    -- zones 0..200 along the south-most row of the grid (west to east)
    -- zones 201..400 along the 2nd-south-most row of the grid
    -- ...
    -- zones 18,001..20,000 along the north-most row of the grid
    -- Lower lat/long bounds are inclusive and upper bounds are exclusive.
    zone_id INT NOT NULL,

    -- Total number of pickups.
    num_pickups INT NOT NULL,
    UNIQUE(zone_id, start_datetime)
);

INSERT INTO pickups_aggregated_manhattan_temp
SELECT
    DATE_FORMAT(MIN(pickup_datetime), '%Y-%m-%d %H:00:00') as start_datetime,
    (
        (FLOOR(pickup_latitude * 100)  - 40 * 100) * 200 +
        (FLOOR(pickup_longitude * 100) + 75 * 100)
    ) as zone_id,
    count(id) as num_pickups
FROM trip_data
WHERE
    -- Rectangle that encloses the whole of NYC. (pickups_aggregated)
    -- pickup_latitude BETWEEN 40.5 AND 41.0 AND
    -- pickup_longitude BETWEEN -74.3 AND -73.7

    -- Rectangle that encloses Manhattan, Brooklyn, Bronx, Queens. (pickups_aggregated_zoomed)
    -- pickup_latitude BETWEEN 40.6 AND 40.9 AND
    -- pickup_longitude BETWEEN -74.0 AND -73.8

    -- Rectangle that encloses Manhattan, part of Queens, Bronx. (pickups_aggregated_tiny)
    -- pickup_latitude BETWEEN 40.7 AND 40.9 AND
    -- pickup_longitude BETWEEN -74.0 AND -73.9

    -- Rectangle that encloses Manhattan. (pickups_aggregated_manhattan)
    pickup_latitude BETWEEN 40.7 AND 40.84 AND
    pickup_longitude BETWEEN -74.02 AND -73.89

    -- Corners at 40.70, -74.02; 40.84, -73.89

GROUP BY CONCAT(
    FLOOR(pickup_longitude * 100), '_',
    FLOOR(pickup_latitude * 100), '_',
    DATE_FORMAT(pickup_datetime, '%Y-%m-%d %H:00:00')
);


-- #### Remove rows for zones which have fewer than 30 pickups in the whole month.

DROP TABLE IF EXISTS zones_to_remove;

CREATE TABLE zones_to_remove (
    zone_id INT NOT NULL
);

INSERT INTO zones_to_remove (
    SELECT zone_id FROM pickups_aggregated_manhattan_temp 
    GROUP by zone_id HAVING SUM(num_pickups) < 30
);

DELETE FROM pickups_aggregated_manhattan_temp
WHERE zone_id IN (
    SELECT zone_id FROM zones_to_remove
);

-- #### Next, find all rows with zero pickups and insert
-- them into pickups_aggregated_manhattan_temp. 

-- Create a table containing all 744 time slots.
DROP TABLE IF EXISTS time_slots;

CREATE TABLE time_slots (
    start_datetime DATETIME NOT NULL
);

INSERT INTO time_slots (
    SELECT DISTINCT start_datetime from pickups_aggregated_manhattan_temp
);

-- Find all rows that have zero pickups and insert them into pickups_aggregated_manhattan_temp.
INSERT INTO pickups_aggregated_manhattan_temp
(
    SELECT start_datetime, zone_id, 0
    FROM (select distinct zone_id from pickups_aggregated_manhattan_temp) Z, (select * from time_slots) T
    WHERE NOT EXISTS (select * from pickups_aggregated_manhattan_temp where zone_id = Z.zone_id and start_datetime = T.start_datetime)
);

-- #### Copy pickups_aggregated_manhattan_temp into our final table,
-- pickups_aggregated_manhattan, and order it by start_datetime.
-- DROP TABLE IF EXISTS pickups_aggregated_manhattan;

-- CREATE TABLE pickups_aggregated_manhattan (
--     start_datetime DATETIME NOT NULL,
--     zone_id INT NOT NULL,
--     num_pickups INT NOT NULL
-- );

-- ALTER TABLE pickups_aggregated_manhattan AUTO_INCREMENT=1;

INSERT INTO pickups_aggregated_manhattan (
    SELECT * from pickups_aggregated_manhattan_temp
    ORDER BY start_datetime
) ORDER BY start_datetime;

-- ALTER TABLE `pickups_aggregated_manhattan` ADD `id` INT NOT NULL AUTO_INCREMENT PRIMARY
-- KEY FIRST;


-- ALTER TABLE pickups_aggregated_manhattan ADD CONSTRAINT unique_zones_times UNIQUE(zone_id, start_datetime);
