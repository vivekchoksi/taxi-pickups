-- File: load_aggregated_from_csv.sql
-- -------------------------------------------
-- Load the data for aggregated pickups from a csv file.

USE taxi_pickups;

DROP TABLE IF EXISTS pickups_aggregated_manhattan;

CREATE TABLE pickups_aggregated_manhattan (
    start_datetime DATETIME NOT NULL,
    zone_id INT NOT NULL,
    num_pickups INT NOT NULL
);

ALTER TABLE pickups_aggregated_manhattan AUTO_INCREMENT=1;

-- Load the data from csv. Modify this path to be the path to
-- the csv data file.
LOAD DATA LOCAL INFILE '/absolute/path/to/repo/data/pickups_aggregated_manhattan.csv'
INTO TABLE pickups_aggregated_manhattan
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

-- Add an id field as the first column in the table.
ALTER TABLE  `pickups_aggregated_manhattan` ADD  `id` INT NOT NULL AUTO_INCREMENT PRIMARY KEY FIRST;