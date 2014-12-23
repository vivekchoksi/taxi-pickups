-- File: load-trip-data.sql
-- --------------------------
-- Load raw trip data into a MySQL table. These queries only load the trip data
-- from one file, e.g. trip_data_1.csv.

DROP TABLE IF EXISTS trip_data;

CREATE TABLE trip_data (
  medallion VARCHAR(63) NOT NULL,
  hack_licence VARCHAR(63) NOT NULL,
  vendor_id VARCHAR(7) NOT NULL,
  rate_code TINYINT NOT NULL,
  store_and_fwd_flag VARCHAR(7) NOT NULL,
  pickup_datetime DATETIME NOT NULL,
  dropoff_datetime DATETIME NOT NULL,
  passenger_count TINYINT NOT NULL,
  trip_time_in_secs INT NOT NULL,
  trip_distance DOUBLE NOT NULL,
  pickup_longitude DOUBLE NOT NULL,
  pickup_latitude DOUBLE NOT NULL,
  dropoff_longitude DOUBLE NOT NULL,
  dropoff_latitude DOUBLE NOT NULL
);

ALTER TABLE trip_data AUTO_INCREMENT=1;

-- Load the data from csv.
LOAD DATA LOCAL INFILE '/absolute/path/to/raw/data/csv'
INTO TABLE trip_data 
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

-- Add an `id` field as the first column in the table.
ALTER TABLE  `trip_data` ADD  `id` INT NOT NULL AUTO_INCREMENT PRIMARY KEY FIRST;
