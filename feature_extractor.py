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
    Note that x is modified to include additional key-value pairs.

    :param x: dict mapping column name to column value for a particular row.

    :return: phi(x) - a dict mapping feature names to feature values.
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
    Learn a list of feature name -> indices mappings in order to vectorize the
    dict feature vector into a numpy array or scipy sparse matrix and return the
    array or sparse matrix ready to be used by an sklearn ML model.

    :param X: list of dicts where each dict maps column names to column values. 
        X is basically the entire training data.

    :return the scipy sparse matrix representing the training data.
    """
    return VECTORIZER.fit_transform([_getFeatureDict(x) for x in X])
