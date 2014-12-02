-- File: add_zero_rows.sql
-- ---------------------------------
-- SQL query to add rows to `pickups_aggregated` where the number of pickups
-- for a zone and hour of day are zero.

-- NOTE: This script is still in progress!

USE taxi_pickups;


-- TODO: Create table called hours.

INSERT INTO pickups_aggregated
(
	SELECT start_datetime, zone_id, 0
	from (
		select *
		from (select distinct zone_id from pickups_aggregated) Z, (select * from hours) H
		where not exists (select * from pickups_aggregated where zone_id = Z.zone_id and HOUR(start_datetime) = H.hour)
	)
);

-- TODO: Order by hour
-- TODO: Re-set all IDs 



