#!/usr/bin/python

import sys
import MySQLdb
from sklearn.metrics import mean_squared_error
from math import sqrt
from models import *

class Database(object):

    def __init__(self):
        self.db = MySQLdb.connect(
            host="localhost", user="root", passwd="",  db=Const.DATABASE_NAME)

    def execute_query(self, query_string, fetch_all=True):
        '''
        :param query_string: sql query as a query_string
        :param fetch_all: True if you want all the results or false if you want
            only one result

        :return: list of rows each represented as a dict - a mapping from column
            names to values. Column names will be what you name columns in case 
            of derived columns such as avg() (using the 'as' keyword in sql)
        '''
        cursor = self.db.cursor()
        cursor.execute(query_string)
        self.db.commit()
        if fetch_all:
            tuple_results = cursor.fetchall()
        else:
            tuple_results = [cursor.fetchone()]
            if (None,) in tuple_results: # get rid of NULL result
                tuple_results.remove((None,))
            elif None in tuple_results: # in case there was no result
                tuple_results.remove(None)
        results = []
        for i, row_tuple in enumerate(tuple_results):
            results.append({
                col_tuple[0]: row_tuple[x] \
                for x, col_tuple in enumerate(cursor.description)
            })
        return results

# The `Dataset` class interfaces with the data.
class Dataset(object):
    '''
    This class assumes that the mysql table used as the dataset is sorted such
    that training examples always come before (i.e. have smaller ids) the test 
    examples in the sorted order.

    Usage:
        dataset = Dataset(0.7, 20) # 14 examples in train set, 6 in test set
        while dataset.hasMoreTrainExamples():
            train_examples = dataset.getTrainExamples(batch_size=2)
            # Do something with the training examples...

        while dataset.hasMoreTestExamples():
            test_example = dataset.getTestExample()
            # Do something with the test example...
    '''

    def __init__(self, train_fraction, dataset_size, database, table_name):
        self.db = database
        self.table_name = table_name # table to read examples from
        self.trainingExamplesLeft = int(train_fraction * dataset_size)
        self.testingExamplesLeft = dataset_size - self.trainingExamplesLeft
        self.last_train_id = self._getLastTrainID()
        self.last_fetched_id = 0

    def hasMoreTrainExamples(self):
        return self.trainingExamplesLeft > 0

    def hasMoreTestExamples(self):
        return self.testingExamplesLeft > 0

    def switchToTestMode(self):
        self.last_fetched_id = self.last_train_id

    def getTrainExamples(self, batch_size=1):
        '''
        :param batch_size: number of training examples to return
        :return: training examples represented as a list of dicts. These may be
            fewer than batch_size in case there are no more training examples.
        '''
        if not self.hasMoreTrainExamples():
            raise Exception("No more training examples left.")
        if batch_size > self.trainingExamplesLeft:
            batch_size = self.trainingExamplesLeft

        examples = self._getExamples(batch_size)
        self.trainingExamplesLeft -= batch_size
        return examples

    def getTestExample(self):
        '''
        :return: test example, represented as a dict.
        '''
        if not self.hasMoreTestExamples():
            raise Exception("No more test examples left.")

        if self.last_fetched_id < self.last_train_id:
            self.switchToTestMode()

        example = self._getExamples(num_examples=1)[0]
        self.testingExamplesLeft -= 1
        return example

    def _getExamples(self, num_examples=1):
        '''
        :param start_id: id of first row to fetch
        :param num_examples: number of examples to return
        :return: examples (i.e. rows) from the data table represented as a dicts
            that map column names to column values
        '''
        query_string = ("SELECT * FROM %s WHERE id > %d limit %d") \
                        % (self.table_name, self.last_fetched_id, num_examples)

        results = self.db.execute_query(query_string)
        self.last_fetched_id = results[len(results) - 1]['id']
        return results

    def _getLastTrainID(self):
        query_string = ("SELECT MAX(id) as max_id FROM "
                        "(SELECT id FROM %s LIMIT %d) T") \
                        % (self.table_name, self.trainingExamplesLeft)

        return self.db.execute_query(query_string, fetch_all=False)[0]['max_id']

    def __str__(self):
        info = 'Num Training Examples: %d, Num Testing Examples: %d' % \
            (self.trainingExamplesLeft, self.testingExamplesLeft)
        info += ('\nFrom table: %s' % self.table_name)
        return info

# The `Evaluator` class evaluates a trained model.
class Evaluator(object):

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def evaluate(self):
        '''
        Evaluate the model on a test set, and print out relevant statistics
        (e.g. RMSD).
        '''

        # Generate a predicted number of pickups for every example in the test
        # data.
        predicted_num_pickups = []
        true_num_pickups = []
        self.dataset.switchToTestMode()
        while self.dataset.hasMoreTestExamples():
            test_example = self.dataset.getTestExample()
            predicted_num_pickups.append(self.model.predict(test_example))
            true_num_pickups.append(test_example['num_pickups'])

        # Evaluate the predictions.
        self.evaluatePredictions(true_num_pickups, predicted_num_pickups)

    def evaluatePredictions(self, true_num_pickups, predicted_num_pickups):
        '''
        Prints some metrics on how well the model performed, including the RMSD.

        :param predicted_num_pickups: List of predicted num_pickups.
        :param true_num_pickups: List of observed num_pickups.

        '''
        assert(len(true_num_pickups) == len(predicted_num_pickups))

        # Compute the RMSD
        rms = sqrt(mean_squared_error(true_num_pickups, predicted_num_pickups))

        # print 'True number of pickups:\t\t' + str(true_num_pickups)
        # print 'Predicted number of pickups:\t' + str(predicted_num_pickups)

        print 'RMSD: %f' % rms


def getModel(model_name, database, dataset):
    lower_name = model_name.lower()
    if lower_name == 'baseline':
        return Baseline(database, dataset)
    elif lower_name == 'betterbaseline':
        return BetterBaseline(database, dataset)
    elif lower_name == 'linear':
        return LinearRegression(database, dataset)
    raise Exception("No model with name %s" % model_name)

def main(args):
    if len(args) < 2:
        print 'Usage: taxi_pickups.py model'
        exit(1)

    database = Database()
    # dataset = Dataset(0.7, Const.DATASET_SIZE, database, Const.AGGREGATED_PICKUPS)
    dataset = Dataset(0.7, 40000, database, Const.AGGREGATED_PICKUPS)
    print dataset
    # Instantiate the specified learning model.
    model = getModel(args[1], database, dataset)
    evaluator = Evaluator(model, dataset)

    # Train the model.
    model.train()

    # Evaluate the model on data from the test set.
    evaluator.evaluate()

if __name__ == '__main__':
    main(sys.argv)
