#!/usr/bin/python

from optparse import OptionParser
import matplotlib.pyplot as plt
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

        num_bins = 100
        plt.hist(num_pickups, num_bins, alpha=0.5)
        plt.yscale('log')
        plt.legend()

        # Label histogram.
        plt.title('Histogram of the number of taxi pickups')
        plt.xlabel('Number of taxi pickups in any zone and hour-long time slot')
        plt.ylabel('Frequency')

        plt.show()


def main(args):
    database = taxi_pickups.Database()
    plotter = Plotter(database, Const.AGGREGATED_PICKUPS)
    plotter.plotNumPickups()

if __name__ == '__main__':
    main(sys.argv)