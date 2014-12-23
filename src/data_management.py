#!/usr/bin/python

import MySQLdb
from passwords import Passwords
from const import Const

class Database(object):

    def __init__(self, is_local):
        '''
        :param is_local: boolean indicating whether to use a local MySQL
                database rather than a remote one.
        '''
        self.is_local = is_local
        self._connect()

    def _connect(self):
        '''
        Connect to the MySQL database.
        '''
        if self.is_local:
            self.db = MySQLdb.connect(
                host=Passwords.LOCAL_MYSQL_HOST,
                user=Passwords.LOCAL_MYSQL_USER,
                passwd=Passwords.LOCAL_MYSQL_PASSWORD,
                db=Const.DATABASE_NAME
            )
        else:
            self.db = MySQLdb.connect(
                host=Passwords.REMOTE_MYSQL_HOST,
                user=Passwords.REMOTE_MYSQL_USER,
                passwd=Passwords.REMOTE_MYSQL_PASSWORD,
                db=Const.DATABASE_NAME
            )

    def execute_query(self, query_string, fetch_all=True):
        '''
        :param query_string: sql query as a query_string
        :param fetch_all: True if you want all the results or false if you want
            only one result

        :return: list of rows each represented as a dict mapping from column
            names to values. Column names will be what you name columns in case
            of derived columns such as avg() (using the 'as' keyword in sql)
        '''

        # Execute the query.
        cursor = None
        try:
            cursor = self.db.cursor()
            cursor.execute(query_string)
        except (AttributeError, MySQLdb.OperationalError):
            # Reconnect to the database if necessary.
            self._connect()
            cursor = self.db.cursor()
            cursor.execute(query_string)

        self.db.commit()

        # Fetch results.
        if fetch_all:
            tuple_results = cursor.fetchall()
        else:
            tuple_results = [cursor.fetchone()]
            # Get rid of NULL result in result set.
            if (None,) in tuple_results:
                tuple_results.remove((None,))
            elif None in tuple_results:
                tuple_results.remove(None)

        # Aggregate and return results.
        results = []
        for i, row_tuple in enumerate(tuple_results):
            results.append({
                col_tuple[0]: row_tuple[x]
                    for x, col_tuple in enumerate(cursor.description)
            })
        return results

# The Dataset class interfaces with the data.
class Dataset(object):
    '''
    This class assumes that the MySQL table used as the dataset is sorted such
    that training examples always come before (i.e. have lesser ids than) the
    test examples in the sorted order.

    Usage:
        dataset = Dataset(0.7, 20) # 14 examples in train set, 6 in test set
        while dataset.hasMoreTrainExamples():
            train_examples = dataset.getTrainExamples(batch_size=2)
            # Do something with the training examples...

        dataset.switchToTestMode()

        while dataset.hasMoreTestExamples():
            test_examples = dataset.getTestExamples(Const.TEST_BATCH_SIZE)
            # Do something with the test example...
    '''

    def __init__(self, train_fraction, dataset_size, max_train_examples, database, table_name):
        self.db = database
        self.table_name = table_name # Table from which to read examples.
        self.trainingExamplesLeft = int(train_fraction * dataset_size)
        self.testingExamplesLeft = dataset_size - self.trainingExamplesLeft
        self.last_train_id = self._getLastTrainID()
        self.last_fetched_id = 0
        self.trainingExamplesLeft = min(self.trainingExamplesLeft, max_train_examples)

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
            raise Exception('No more training examples left.')
        if batch_size > self.trainingExamplesLeft:
            batch_size = self.trainingExamplesLeft

        examples = self._getExamples(batch_size)
        self.trainingExamplesLeft -= batch_size
        return examples

    def getTestExamples(self, batch_size=1):
        '''
        :return: test example, represented as a dict.
        '''
        if not self.hasMoreTestExamples():
            raise Exception('No more test examples left.')

        if self.last_fetched_id < self.last_train_id:
            self.switchToTestMode()

        if batch_size > self.testingExamplesLeft:
            batch_size = self.testingExamplesLeft

        examples = self._getExamples(batch_size)
        self.testingExamplesLeft -= batch_size
        return examples

    def _getExamples(self, num_examples=1):
        '''
        :param start_id: id of first row to fetch
        :param num_examples: number of examples to return
        :return: examples (i.e. rows) from the data table represented as a dicts
            that map column names to column values
        '''
        query_string = ('SELECT * FROM %s WHERE id > %d limit %d') \
                        % (self.table_name, self.last_fetched_id, num_examples)

        results = self.db.execute_query(query_string)
        self.last_fetched_id = results[len(results) - 1]['id']
        return results

    def _getLastTrainID(self):
        query_string = ('SELECT MAX(id) as max_id FROM '
                        '(SELECT id FROM %s LIMIT %d) T') \
                        % (self.table_name, self.trainingExamplesLeft)

        return self.db.execute_query(query_string, fetch_all=False)[0]['max_id']

    def __str__(self):
        info = '%d training examples, %d testing examples ' % \
            (self.trainingExamplesLeft, self.testingExamplesLeft)
        info += ('from table: %s' % self.table_name)
        return info