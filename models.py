#!/usr/bin/python
import MySQLdb
import datetime
import os
from abc import ABCMeta, abstractmethod
from const import Const

# Interface for our learning models.
class Model(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, dataset):
        '''
        Trains the learning model on the list of training examples provided in
        the dataset.
        '''
        pass

    @abstractmethod
    def predict(self, test_example):
        '''
        Predicts the number of pickups for the test example provided.

        :param test_example: dict mapping feature names to feature values

        :return: Predicted number of pickups for the test example.
        '''
        pass

    @abstractmethod
    def generateTestData(self):
        '''
        Generates the data we use to evaluate our learning models.

        :return: (test_data, true_num_pickups), where
            test_data is a list of tuples of the form 
            (pickup_datetime, pickup_lat, pickup_long), and true_num_pickups is 
            a list of the actual number of pickups observed (to be used to 
            evaluate the predictions).
        '''
        pass

class Baseline(Model):

    def __init__(self, database):
        self.db = database
        self.table_name = Const.AGGREGATED_PICKUPS

    def train(self, dataset):
        '''
        The SQL script to generate the aggregated pickups table is commented out
        because we only need to run it once.

        See Model for comments on the parameters and return value.
        '''
        # Note: this line of code isn't tested yet.
        # os.system('mysql -u root < pickups-aggregated.sql')
        pass

    def predict(self, test_example):
        '''
        Predicts the number of pickups at the specified time and location, 
        within a 1 hour interval and 0.01 x 0.01 degrees lat/long box.

        See Model for comments on the parameters and return value.
        '''
        num_pickups = None
        pickup_time, pickup_lat, pickup_long = test_example
        query_string = "SELECT AVG(num_pickups) FROM %s WHERE " \
                        "HOUR(start_datetime) = %d AND " \
                        "zone_id = %d" \
                        % (self.table_name, pickup_time.hour, zone_id)
        # print "Querying 'trip_data': " + query_string
        row = self.db.execute_query(query_string)
        if row is not None and row[0] is not None:
            num_pickups = float(row[0])
        else:
            num_pickups = 0.0

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