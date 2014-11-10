-- File: pickups-aggregated.sql
-- --------------------------
-- SQL query to select a table with taxi
-- pickup counts aggregated by time and location
-- windows.


-- Runs in < 60 seconds on the first trip_data table.

-- TODO: Create a table with the results of this query.

SELECT FLOOR(pickup_longitude * 100) / 100 as longitude, FLOOR(pickup_latitude * 100) / 100 as latitude, HOUR(pickup_datetime), count(id) as aggregate_count
FROM trip_data
WHERE
  pickup_latitude BETWEEN 40 AND 41 AND
  pickup_longitude BETWEEN -75 AND -73
GROUP BY (
 FLOOR(pickup_longitude * 100) * 1000000 +
 FLOOR(pickup_latitude * 100) *  100 +
 HOUR(pickup_datetime)
);


-- DROP TABLE IF EXISTS pickups_aggregated;

-- CREATE TABLE pickups_aggregated (
--   id int AUTO_INCREMENT NOT NULL,
--   start_datetime DATETIME NOT NULL,
--   end_datetime DATETIME NOT NULL,
--   start_longitude DOUBLE NOT NULL,
--   end_longitude DOUBLE NOT NULL,
--   start_latitude DOUBLE NOT NULL,
--   end_latitude DOUBLE NOT NULL,
--   num_pickups INT NOT NULL
-- );
