-- File: pickups-aggregated.sql
-- --------------------------
-- SQL query to select a table with taxi
-- pickup counts aggregated by time and location
-- windows.


-- Runs in < 60 seconds on the first trip_data table.

-- TODO: Create a table with the results of this query.
USE taxi_pickups;

DROP TABLE IF EXISTS zone;

-- Lower bound is inclusive and upper bound is exclusive
CREATE TABLE zone (
    min_long DOUBLE NOT NULL,
    min_lat DOUBLE NOT NULL
);

ALTER TABLE zone AUTO_INCREMENT=1;

INSERT INTO zone
SELECT DISTINCT
    FLOOR(pickup_longitude * 100) / 100 as min_long, 
    FLOOR(pickup_latitude * 100) / 100 as min_lat
FROM trip_data
WHERE
    pickup_latitude BETWEEN 40 AND 41 AND
    pickup_longitude BETWEEN -75 AND -73
;

ALTER TABLE `zone` ADD `id` INT NOT NULL AUTO_INCREMENT PRIMARY KEY FIRST;

DROP TABLE IF EXISTS pickups_aggregated;

CREATE TABLE pickups_aggregated (
    -- fixed time interval width so no need for end time
    start_datetime DATETIME NOT NULL,
    zone_id INT references zone(id),
    num_pickups INT NOT NULL
);

ALTER TABLE pickups_aggregated AUTO_INCREMENT=1;

INSERT INTO pickups_aggregated
SELECT
    DATE_FORMAT(MIN(pickup_datetime), '%Y-%m-%d %H:00:00') as start_datetime,
    (
        SELECT id FROM zone 
        WHERE zone.min_lat = (FLOOR(pickup_latitude * 100) / 100) and 
                zone.min_long = (FLOOR(pickup_longitude * 100) / 100)
    ),
    count(id) as num_pickups
FROM trip_data
WHERE
    pickup_latitude BETWEEN 40 AND 41 AND
    pickup_longitude BETWEEN -75 AND -73
GROUP BY CONCAT( 
    FLOOR(pickup_longitude * 100), '_',
    FLOOR(pickup_latitude * 100), '_',
    DATE_FORMAT(pickup_datetime, '%Y-%m-%d %H:00:00')
);

ALTER TABLE `pickups_aggregated` ADD `id` INT NOT NULL AUTO_INCREMENT PRIMARY 
KEY FIRST;