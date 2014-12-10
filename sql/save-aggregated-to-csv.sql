-- File: save-aggregated-to-csv.sql
-- --------------------------
-- This script saves the pickups_aggregated table to a .csv file named
-- pickups-aggregated.csv that is stored in the root mysql data
-- directory. 
-- (For my Mac, this is /usr/local/mysql/data/taxi_pickups/pickups-aggregated_manhattan.csv)

SELECT *
FROM pickups_aggregated_manhattan
INTO OUTFILE 'pickups-aggregated_manhattan.csv'
FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n';
