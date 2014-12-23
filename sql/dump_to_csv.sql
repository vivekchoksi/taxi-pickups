-- File: dump_to_csv.sql
-- --------------------------
-- Save the pickups_aggregated_manhattan table to a .csv file named
-- pickups_aggregated_manhattan.csv that is stored in the root mysql data
-- directory.
--
-- NOTE(vivekchoksi):
-- For my Mac, the data is written to /usr/local/mysql/data/taxi_pickups/pickups_aggregated_manhattan.csv

SELECT *
FROM pickups_aggregated_manhattan
INTO OUTFILE 'pickups_aggregated_manhattan.csv'
FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n';
