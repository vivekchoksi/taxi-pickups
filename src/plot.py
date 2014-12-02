#!/usr/bin/python

from optparse import OptionParser
import matplotlib.pyplot as plt
import sys
import taxi_pickups
from const import Const

class Plotter(object):

    def __init__(self, database, table_name):
        self.db = database

        query_string = ('SELECT * FROM %s limit %d') \
                        % (table_name, 100)
        self.data = self.db.execute_query(query_string)

    def plotNumPickups(self):
        num_pickups = []
        for row in self.data:
            num_pickups.append(row['num_pickups'])
        plt.hist(num_pickups)
        plt.show()


def main(args):
    database = taxi_pickups.Database()
    plotter = Plotter(database, Const.AGGREGATED_PICKUPS)
    plotter.plotNumPickups()

if __name__ == '__main__':
    main(sys.argv)