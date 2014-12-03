#!/usr/bin/python

from optparse import OptionParser
import matplotlib.pyplot as plt
import numpy as np
import sys
import taxi_pickups
from const import Const

class Plotter(object):

    def __init__(self, database, table_name):
        self.db = database
        num_examples = Const.DATASET_SIZE
        query_string = ('SELECT * FROM %s limit %d') \
                        % (table_name, num_examples)
        self.data = self.db.execute_query(query_string)

    def plotNumPickups(self):
        '''
        Plot a histogram showing the distribution of true number of pickups.
        '''
        # Get data into array.
        num_pickups = []
        for row in self.data:
            num_pickups.append(row['num_pickups'])

        # Plot histogram.
        num_bins = 100
        plt.hist(num_pickups, num_bins, alpha=0.5)
        plt.yscale('log')
        plt.legend()

        # Label histogram.
        plt.title('Histogram of the number of taxi pickups')
        plt.xlabel('Number of taxi pickups in any zone and hour-long time slot')
        plt.ylabel('Frequency')

        plt.show()

    def plotNumPickupsByDay(self):
        '''
        Plot a histogram showing the distribution of true number of pickups.
        '''
        # Get data into arrays.
        num_days_in_week = 7
        days_in_week = np.array(range(num_days_in_week))
        num_pickups = [0]*num_days_in_week

        for row in self.data:
            day = row['start_datetime'].weekday()
            num_pickups[day] += 1

        # Plot bar chart.
        width = 1
        fig, ax = plt.subplots()
        ax.bar(days_in_week + width / 2.0, num_pickups, width=width, alpha=0.5)
        ax.set_xticks(days_in_week+width)

        # Label bar chart.
        plt.title('Number of taxi pickups by day of week')
        plt.xlabel('Day of week')
        plt.ylabel('Number of taxi pickups')
        ax.set_xticklabels(('Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'))

        plt.show()


    def plotNumPickupsByHour(self):
        '''
        Plot a histogram showing the distribution of true number of pickups.
        '''
        # Get data into arrays.
        num_hours = 24
        hours_in_day = np.array(range(num_hours))
        num_pickups = [0]*num_hours

        for row in self.data:
            hour = row['start_datetime'].hour
            num_pickups[hour] += 1

        # Plot bar chart.
        # FIXME: X label and tick mark positioning.
        width = 1
        fig, ax = plt.subplots()
        ax.bar(hours_in_day - width / 2.0, num_pickups, width=width, alpha=0.5)
        ax.set_xticks(hours_in_day+width)

        # Label bar chart.
        plt.title('Number of taxi pickups by hour of day')
        plt.xlabel('Hour of day')
        plt.ylabel('Number of taxi pickups')
        x_labels = hours_in_day.copy().tolist()
        for idx, hour in enumerate(x_labels):
            suffix = 'am' if hour < 12 else 'pm'
            hour = hour % 12
            x_labels[idx] = '' if hour % 2 == 0 else str(hour) + suffix
        ax.set_xticklabels(x_labels)

        plt.show()

    def plotNumPickupsByZone(self):
        '''
        Plot a histogram showing the distribution of true number of pickups by zone.
        Also, print out the number of zones with fewer than some threshold of
        pickups during the entire month.
        '''
        # Gather data.
        num_pickups_by_zone = {}
        num_pickups_list = []

        pickups_threshold = 5
        num_under_threshold = 0

        for row in self.data:
            zone_id = str(row['zone_id'])
            num_pickups_by_zone[zone_id] = num_pickups_by_zone.get(zone_id, 0) + row['num_pickups']

        for num_pickups in num_pickups_by_zone.values():
            num_pickups_list.append(num_pickups)
            if num_pickups < pickups_threshold:
                num_under_threshold += 1

        print len(num_pickups_by_zone), 'total zones'
        print num_under_threshold, 'zones for which there were fewer than', pickups_threshold, \
            'pickups during the entire month'

        # Plot histogram.
        num_bins = 100
        plt.hist(num_pickups_list, num_bins, alpha=0.5)

        # Label histogram.
        plt.title('Histogram of the number of taxi pickups by zone')
        plt.xlabel('Number of taxi pickups in a zone during one month')
        plt.ylabel('Frequency')

        plt.show()

def main(args):
    database = taxi_pickups.Database()
    plotter = Plotter(database, Const.AGGREGATED_PICKUPS)
    # plotter.plotNumPickups()
    # plotter.plotNumPickupsByDay()
    # plotter.plotNumPickupsByHour()
    plotter.plotNumPickupsByZone()

if __name__ == '__main__':
    main(sys.argv)
