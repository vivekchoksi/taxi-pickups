#!/usr/bin/python

import ConfigParser
import math
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import MiniBatchKMeans
from weather import Weather

CONFIG = ConfigParser.RawConfigParser()
CONFIG.read('features.cfg')
FEATURE_SELECTION = 'FeatureSelection'
VECTORIZER = None   # Will be initialized later.
PRECLUSTER_VECTORIZER = DictVectorizer(sparse=True) # Vectorizer without cluster features.
CLUSTERER = MiniBatchKMeans(n_clusters=15, init='k-means++')
WEATHER = None      # Will be initialized later.

def _extractZone(x, feature_dict):
    feature_dict['Zone'] = str(x['zone_id'])

def _extractHourOfDay(x, feature_dict):
    feature_dict['HourOfDay'] = str(x['start_datetime'].hour)

def _extractDayOfWeek(x, feature_dict):
    feature_dict['DayOfWeek'] = str(x['start_datetime'].weekday())

def _extractDayOfMonth(x, feature_dict):
    feature_dict['DayOfMonth'] = str(x['start_datetime'].day)

def _extractZoneHourOfDay(x, feature_dict):
    feature_dict['ZoneHourOfDay'] = str(x['zone_id']) + '_' + str(x['start_datetime'].hour)

def _extractCluster(x, feature_dict):
    feature_dict['Cluster'] = str(x['cluster_id'])

def _extractWeather(x, feature_dict):
    daily_weather = WEATHER.getWeather(x['start_datetime'])
    _extractRainfall(daily_weather['PRCP'], feature_dict)

def _extractRainfall(rainfall, feature_dict):
    if rainfall == 0:
        feature_dict['Rainfall'] = 0
    elif rainfall < 10:
        feature_dict['Rainfall'] = 1
    else:
        feature_dict['Rainfall'] = 2

def _appendClusterFeatures(feature_dicts, is_test):
    """
    If training, first runs k-means to compute the optimal centroids.
    For each example in feature_dicts, appends the index of the nearest centroid
    as a feature.

    :param feature_dict: feature dictionary computed using _getFeatureDict().
    :param is_test: whether extracting features for testing or training purposes.
    """
    global PRECLUSTER_VECTORIZER
    X_vectors = None
    if not is_test:
        X_vectors = PRECLUSTER_VECTORIZER.fit_transform(feature_dicts)
        CLUSTERER.fit(X_vectors)
    else:
        X_vectors = PRECLUSTER_VECTORIZER.transform(feature_dicts)

    Z = CLUSTERER.predict(X_vectors)
    [_extractCluster({'cluster_id': Z[i]}, feature_dicts[i]) for i in xrange(len(Z))]

def _getFeatureDict(x):
    """
    Transform a training or testing example into a feature vector.

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
    if CONFIG.getboolean(FEATURE_SELECTION, 'ZoneHourOfDay'):
        _extractZoneHourOfDay(x, feature_dict)
    if CONFIG.getboolean(FEATURE_SELECTION, 'Weather'):
        _extractWeather(x, feature_dict)
    return feature_dict

def getFeatureVectors(X, use_sparse, is_test=False):
    """
    Transform the input list of training examples from a list of dicts to a
    numpy array or scipy sparse matrix for input into an sklearn ML model.

    :param X: a list of training examples, represented as a list of dicts where
              each dict maps column names to column values.

    :param use_sparse: boolean for whether the return value should be
                represented as a sparse matrix.

    :return: the scipy matrix that represents the training data.
    """
    global VECTORIZER
    if VECTORIZER is None:
        VECTORIZER = DictVectorizer(sparse=use_sparse)

    global WEATHER
    if CONFIG.getboolean(FEATURE_SELECTION, 'Weather'):
        WEATHER = Weather()

    feature_dicts = [_getFeatureDict(x) for x in X]

    # If clustering is turned on, compute the centroids, then
    # append the nearest centroid ID to each feature vector.
    if CONFIG.getboolean(FEATURE_SELECTION, 'Cluster'):
        _appendClusterFeatures(feature_dicts, is_test)

    transformed = VECTORIZER.transform(feature_dicts) \
        if is_test \
        else VECTORIZER.fit_transform(feature_dicts)

    return transformed

def getFeatureNameIndices():
    """
    Use this to know which indices in the sklearn vectors correspond
    to which features.

    :return: dict that maps feature names to indices.
    """
    return VECTORIZER.vocabulary_