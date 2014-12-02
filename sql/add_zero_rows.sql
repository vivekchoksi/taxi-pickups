-- File: add_zero_rows.sql
-- ---------------------------------
-- SQL query to add rows to `pickups_aggregated_temp` where the number of pickups
-- for a zone and hour of day are zero.

-- NOTE: This script is still in progress!

USE taxi_pickups;


-- Create a table containing all 744 time slots.
DROP TABLE IF EXISTS time_slots;

CREATE TABLE time_slots (
    start_datetime DATETIME NOT NULL
);

INSERT INTO time_slots (
	SELECT DISTINCT start_datetime from pickups_aggregated_temp
);


-- Find all rows that have zero pickups and insert them into pickups_aggregated_temp.
INSERT INTO pickups_aggregated_temp
(
	SELECT start_datetime, zone_id, 0		
	FROM (select distinct zone_id from pickups_aggregated_temp) Z, (select * from time_slots) T
	WHERE NOT EXISTS (select * from pickups_aggregated_temp where zone_id = Z.zone_id and start_datetime = T.start_datetime)
);




