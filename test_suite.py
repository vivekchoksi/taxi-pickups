#!/usr/bin/python

from taxi_pickups import *
import unittest

class TestDataset(unittest.TestCase):

    def setUp(self):
        self.db = Database()

    def test_ids_0_7_20(self):
        self._train_ids_one_by_one(0.7, 20, 14, 6, 1, 1)

    def test_ids_0_1_10(self):
        self._train_ids_one_by_one(0.1, 10, 1, 9, 1, 1)

    def test_ids_0_75_10(self):
        self._train_ids_one_by_one(0.75, 10, 7, 3, 1, 1)

    def test_ids_batch_0_75_10(self):
        self._train_ids_one_by_one(0.75, 10, 7, 3, 2, 1)

    def test_ids_batch_0_75_10_5(self):
        self._train_ids_one_by_one(0.75, 10, 7, 3, 5, 2)

    def _train_ids_one_by_one(self, train_fraction, dataset_size, 
        num_training_examples, num_test_examples, batch_size, last_batch_size):

        data = Dataset(
            train_fraction, dataset_size, self.db, Const.TRIP_DATA)
        count = 1
        last = False
        while data.hasMoreTrainExamples():
            examples = data.getTrainExamples(batch_size)
            if last_batch_size != batch_size and \
                len(examples) == last_batch_size:
                self.assertFalse(last)
                last = True
            else:
                self.assertEquals(len(examples), batch_size)
            for example in examples:
                self.assertEqual(example['id'], count)
                count += 1

        self.assertEqual(count, num_training_examples + 1)

        while data.hasMoreTestExamples():
            example = data.getTestExample()
            self.assertEqual(example['id'], count)
            count += 1

        self.assertEqual(count, num_training_examples + num_test_examples + 1)

if __name__ == '__main__':
    unittest.main()