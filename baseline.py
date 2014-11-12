#!/usr/bin/python
import MySQLdb
import datetime
import os

import taxi_pickups

class Baseline(taxi_pickups.Model):
    def train(self, dataset):
        '''
        The SQL script to generate the aggregated pickups table is commented out
        because we only need to run it once.

        See taxi_pickups.Model for comments on the parameters and return value.
        '''
        # Get training examples from dataset
        # Feed them to model and train the model

        # Note: this line of code isn't tested yet.
        # os.system('mysql -u root < pickups-aggregated.sql')
        pass

    def predict(self, test_data):
        '''
        Predicts the number of pickups at the specified time and location, within a 1 hour interval
        and 0.01 x 0.01 degrees lat/long box.

        See taxi_pickups.Model for comments on the parameters and return value.
        '''
        # Connect to the db.
        db = MySQLdb.connect(host="localhost", user="root", passwd="",  db="taxi_pickups")
        cursor = db.cursor()

        num_pickups = []

        for pickup_time, pickup_lat, pickup_long in test_data:
            # TODO we should put the lat/long ==> zone_id logic EITHER in the database OR in Python.
            zone_id = int(
                            int(round(pickup_lat * 100) - 40 * 100) * 200 + \
                            int(round(pickup_long * 100) + 75 * 100)
                        )
            query_string = "SELECT AVG(num_pickups) FROM pickups_aggregated WHERE " \
                            "HOUR(start_datetime) = %d AND " \
                            "zone_id = %d" \
                            % (pickup_time.hour, zone_id)
            # print "Querying 'trip_data': " + query_string
            cursor.execute(query_string)
            db.commit()
            row = cursor.fetchone()
            if row is not None and row[0] is not None:
                num_pickups.append(float(row[0]))
            else:
                num_pickups.append(0.0)

        return num_pickups

    def generateTestData(self):
        '''
        See taxi_pickups.Model for comments on the parameters and return value.
        '''
        # TODO ********************
        # TODO ** Automate this! **
        # TODO ********************

        # TODO We need to split pickups_aggregated into train, dev, and test sets.
        def zoneIdToLat(zone_id):
            return (int(zone_id) / 200 + 40 * 100) / 100.0

        def zoneIdToLong(zone_id):
            return (int(zone_id) % 200 - 75 * 100) / 100.0

        test_data = [(datetime.datetime(2013, 1, 27, 16, 11, 12, 30), zoneIdToLat(13326), zoneIdToLong(13326)),
                     (datetime.datetime(2013, 1, 1, 2, 11, 12, 30), zoneIdToLat(12922), zoneIdToLong(12922)),
                     (datetime.datetime(2013, 1, 15, 2, 11, 12, 30), zoneIdToLat(20), zoneIdToLong(20))]
        true_num_pickups = [6, 16, 0]

        return test_data, true_num_pickups