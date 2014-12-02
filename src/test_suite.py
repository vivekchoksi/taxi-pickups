#!/usr/bin/python
import unittest
import datetime
import feature_extractor
from taxi_pickups import *

class TestFeatureExtractor(unittest.TestCase):

    def setUp(self):
        self.db = Database()

    def test_time_1(self):
        self._test_time_features('2014-11-22 10:00:00', 10, 5, 22)

    def _test_time_features(self, datetime_str, hourOfDay, dayOfWeek, dayOfMonth):
        x = {
            'start_datetime': datetime.datetime.strptime(
                datetime_str, '%Y-%m-%d %H:%M:%S')
        }
        feature_dict = {}
        feature_extractor._extractHourOfDay(x, feature_dict)
        feature_extractor._extractDayOfWeek(x, feature_dict)
        feature_extractor._extractDayOfMonth(x, feature_dict)
        self.assertEquals(feature_dict['HourOfDay'], hourOfDay)
        self.assertEquals(feature_dict['DayOfWeek'], dayOfWeek)
        self.assertEquals(feature_dict['DayOfMonth'], dayOfMonth)

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
            train_fraction, dataset_size, self.db, Const.AGGREGATED_PICKUPS)
        last_id = 0
        count = 0
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
                self.assertTrue(example['id'] > last_id)
                last_id = example['id']
                count += 1

        self.assertEqual(count, num_training_examples)

        while data.hasMoreTestExamples():
            example = data.getTestExample()
            self.assertTrue(example['id'] > last_id)
            last_id = example['id']
            count += 1

        self.assertEqual(count, num_training_examples + num_test_examples)

class TestBaseline(unittest.TestCase):

    def setUp(self):
        self.db = Database()
        self.model = Baseline(self.db, dataset=Dataset(
            1.0, 73727, self.db, Const.AGGREGATED_PICKUPS))
        self.table_name = Const.AGGREGATED_PICKUPS

    def test_basic_prediction(self):
        query_string = "SELECT * FROM %s WHERE id = %d" \
                        % (self.table_name, 73727)

        test_example = self.db.execute_query(query_string)[0]
        self.assertAlmostEqual(self.model.predict(test_example), 1171.9475)

if __name__ == '__main__':
    unittest.main()