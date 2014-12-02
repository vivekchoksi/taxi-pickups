#!/usr/bin/python
import ConfigParser
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import MiniBatchKMeans
from weather import Weather


FEATURE_SELECTION = 'FeatureSelection'

class FeatureExtractor(object):

    def __init__(self, use_sparse):
        self.config = ConfigParser.RawConfigParser()
        self.config.read('features.cfg')
        self.vectorizer = DictVectorizer(sparse=use_sparse)

        if self.config.getboolean(FEATURE_SELECTION, 'Cluster'):
            self.precluster_vectorizer = DictVectorizer(sparse=use_sparse) # Vectorizer without cluster features.
            self.clusterer = MiniBatchKMeans(n_clusters=15, init='k-means++')
        if self.config.getboolean(FEATURE_SELECTION, 'Weather'):
            self.weather_data = Weather()

    def getFeatureVectors(self, X, is_test=False):
        """
        Transform the input list of training examples from a list of dicts to a
        numpy array or scipy sparse matrix for input into an sklearn ML model.

        :param X: a list of training examples, represented as a list of dicts where
                  each dict maps column names to column values.

        :param use_sparse: boolean for whether the return value should be
                    represented as a sparse matrix.

        :return: the scipy matrix that represents the training data.
        """
        feature_dicts = [self._getFeatureDict(x) for x in X]

        # If clustering is turned on, compute the centroids, then
        # append the nearest centroid ID to each feature vector.
        if self.config.getboolean(FEATURE_SELECTION, 'Cluster'):
            self._appendClusterFeatures(feature_dicts, is_test)

        transformed = self.vectorizer.transform(feature_dicts) \
            if is_test \
            else self.vectorizer.fit_transform(feature_dicts)

        return transformed

    def getFeatureNameIndices(self):
        """
        Use this to know which indices in the sklearn vectors correspond
        to which features.

        :return: dict that maps feature names to indices.
        """
        return self.vectorizer.vocabulary_

    def _extractZone(self, x, feature_dict):
        feature_dict['Zone'] = str(x['zone_id'])

    def _extractHourOfDay(self, x, feature_dict):
        feature_dict['HourOfDay'] = str(x['start_datetime'].hour)

    def _extractDayOfWeek(self, x, feature_dict):
        feature_dict['DayOfWeek'] = str(x['start_datetime'].weekday())

    def _extractDayOfMonth(self, x, feature_dict):
        feature_dict['DayOfMonth'] = str(x['start_datetime'].day)

    def _extractZoneHourOfDay(self, x, feature_dict):
        feature_dict['Zone_HourOfDay'] = str(x['zone_id']) + '_' + str(x['start_datetime'].hour)

    # Concatenates the zone, whether the taxi ride is on a weekend, and the hour
    # of day.
    def _extractZoneWeekendHour(self, x, feature_dict):
        feature_dict['Zone_IsWeekend_Hour'] = str(x['zone_id']) + '_' + \
                                          str(self._isWeekend(x['start_datetime'])) + '_' + \
                                          str(x['start_datetime'].hour)

    # Concatenates the zone, day of week, and hour of day.
    def _extractZoneDayHour(self, x, feature_dict):
        feature_dict['Zone_DayOfWeek_Hour'] = str(x['zone_id']) + '_' + \
                                          str(x['start_datetime'].weekday()) + '_' + \
                                          str(x['start_datetime'].hour)

    def _isWeekend(self, date):
        # Weekday = 0 for Monday.
        return 1 if date.weekday() >= 5 else 0

    def _extractCluster(self, x, feature_dict):
        feature_dict['Cluster'] = str(x['cluster_id'])

    def _extractWeather(self, x, feature_dict):
        daily_weather = self.weather_data.getWeather(x['start_datetime'])
        self._extractRainfall(daily_weather['PRCP'], feature_dict)

    def _extractRainfall(self, rainfall, feature_dict):
        if rainfall == 0:
            feature_dict['Rainfall'] = 0
        elif rainfall < 100:
            feature_dict['Rainfall'] = 1
        else:
            feature_dict['Rainfall'] = 2

    def _appendClusterFeatures(self, feature_dicts, is_test):
        """
        If training, first runs k-means to compute the optimal centroids.
        For each example in feature_dicts, appends the index of the nearest centroid
        as a feature.

        :param feature_dict: feature dictionary computed using _getFeatureDict().
        :param is_test: whether extracting features for testing or training purposes.
        """
        X_vectors = None
        if not is_test:
            X_vectors = self.precluster_vectorizer.fit_transform(feature_dicts)
            self.clusterer.fit(X_vectors)
        else:
            X_vectors = self.precluster_vectorizer.transform(feature_dicts)

        Z = self.clusterer.predict(X_vectors)
        [self._extractCluster({'cluster_id': Z[i]}, feature_dicts[i]) for i in xrange(len(Z))]

    def _getFeatureDict(self, x):
        """
        Transform a training or testing example into a feature vector.

        :param x: a dict representing one row in a data table.
        :return: phi(x), a dict mapping feature names to feature values.
        """
        feature_dict = {}
        if self.config.getboolean(FEATURE_SELECTION, 'Zone'):
            self._extractZone(x, feature_dict)
        if self.config.getboolean(FEATURE_SELECTION, 'HourOfDay'):
            self._extractHourOfDay(x, feature_dict)
        if self.config.getboolean(FEATURE_SELECTION, 'DayOfWeek'):
            self._extractDayOfWeek(x, feature_dict)
        if self.config.getboolean(FEATURE_SELECTION, 'DayOfMonth'):
            self._extractDayOfMonth(x, feature_dict)
        if self.config.getboolean(FEATURE_SELECTION, 'Zone_HourOfDay'):
            self._extractZoneHourOfDay(x, feature_dict)
        if self.config.getboolean(FEATURE_SELECTION, 'Weather'):
            self._extractWeather(x, feature_dict)
        if self.config.getboolean(FEATURE_SELECTION, 'Zone_IsWeekend_Hour'):
            self._extractZoneWeekendHour(x, feature_dict)
        if self.config.getboolean(FEATURE_SELECTION, 'Zone_DayOfWeek_Hour'):
            self._extractZoneDayHour(x, feature_dict)
        return feature_dict