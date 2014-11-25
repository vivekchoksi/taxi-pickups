#!/usr/bin/python

import ConfigParser, datetime
from sklearn.feature_extraction import DictVectorizer

CONFIG = ConfigParser.RawConfigParser()
CONFIG.read('features.cfg')
FEATURE_SELECTION = 'FeatureSelection'
VECTORIZER = DictVectorizer(sparse=True)

def _extractZone(x, feature_dict):
    feature_dict['Zone'] = x['zone_id']

def _extractHourOfDay(x, feature_dict):
    feature_dict['HourOfDay'] = x['start_datetime'].hour

def _extractDayOfWeek(x, feature_dict):
    feature_dict['DayOfWeek'] = x['start_datetime'].weekday()

def _extractDayOfMonth(x, feature_dict):
    feature_dict['DayOfMonth'] = x['start_datetime'].day

def _getFeatureDict(x):
    """
    Transform a training or testing example into a feature vector.
    Note that x is modified to include additional key-value pairs.

    :param x: a dict representing one row in a data table.
    :return: phi(x), a dict mapping feature names to feature values.
    """
    feature_dict = {}
    if CONFIG.getboolean(FEATURE_SELECTION, 'Zone'):
        _extractZone(x, feature_dict)
    if CONFIG.getboolean(FEATURE_SELECTION, 'HourOfDay'):
        _extractHourOfDay(x, feature_dict)
    if CONFIG.getboolean(FEATURE_SELECTION, 'DayOfWeek'):
        _extractDayOfWeek(x, feature_dict)
    if CONFIG.getboolean(FEATURE_SELECTION, 'DayOfMonth'):
        _extractDayOfMonth(x, feature_dict)
    return feature_dict

def getFeatureVectors(X):
    """
    Transform the input list of training examples from a list of dicts to a
    numpy array or scipy sparse matrix for input into an sklearn ML model.

    :param X: a list of training examples, represented as a list of dicts where
              each dict maps column names to column values.

    :return: the scipy sparse matrix that represents the training data.
    """
    return VECTORIZER.fit_transform([_getFeatureDict(x) for x in X])
