#!/usr/bin/python
import ConfigParser
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import MiniBatchKMeans
from weather import Weather
import util

FEATURE_SELECTION = 'FeatureSelection'

class FeatureExtractor(object):

    def __init__(self, use_sparse):
        self.config = ConfigParser.RawConfigParser()
        self.config.read(util.FEATURES_FILE)
        self.vectorizer = DictVectorizer(sparse=use_sparse)

        if self.config.getboolean(FEATURE_SELECTION, 'Cluster'):
            self.precluster_vectorizer = DictVectorizer(sparse=use_sparse) # Vectorizer without cluster features.
            self.clusterer = MiniBatchKMeans(n_clusters=15, init='k-means++')
        if self.config.getboolean(FEATURE_SELECTION, 'DailyWeather') or \
            self.config.getboolean(FEATURE_SELECTION, 'HourlyWeather') or \
            self.config.getboolean(FEATURE_SELECTION, 'Zone_DayOfWeek_HourlyWeather'):
            self.weather_data = Weather()

        if util.VERBOSE:
            feature_list = self.getFeatureList()
            print 'Feature Template List:'
            for feature in feature_list:
                print '\t%s' % feature

    def getFeatureList(self):
        feature_list = []
        for feature in self.config.options(FEATURE_SELECTION):
            if self.config.getboolean(FEATURE_SELECTION, feature):
                feature_list.append(feature)
        return feature_list

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
        feature_dicts = [self.getFeatureDict(x) for x in X]

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
        feature_dict['HourOfDay'] = '%02d' % x['start_datetime'].hour # Pad hours < 10 with a leading zero.

    def _extractDayOfWeek(self, x, feature_dict):
        feature_dict['DayOfWeek'] = '%02d' % x['start_datetime'].weekday() # Pad day of week with a leading zero.

    def _extractZoneHourOfDay(self, x, feature_dict):
        feature_dict['Zone_HourOfDay'] = '%d_%02d' % (x['zone_id'], x['start_datetime'].hour)

    # Concatenates the zone, day of week, and hour of day.
    def _extractZoneDayHour(self, x, feature_dict):
        feature_dict['Zone_DayOfWeek_Hour'] = '%d_%02d_%02d' % (x['zone_id'],
                                                                x['start_datetime'].weekday(), # Pad day of week with a leading zero.
                                                                x['start_datetime'].hour) # Pad hours < 10 with a leading zero.

    def _extractZoneDayOfWeekHourlyWeather(self, x, feature_dict):
        hourly_weather = self.weather_data.getHourlyWeather(x['start_datetime'])
        rainfallValue = self._getHourlyRainfallValue(hourly_weather['PRCP'], True)
        feature_dict['Zone_DayOfWeek_HourlyRainfall'] = '%d_%02d_%s' % (x['zone_id'],
                                                                x['start_datetime'].weekday(),
                                                                rainfallValue)


    def _extractCluster(self, x, feature_dict):
        feature_dict['Cluster'] = str(x['cluster_id'])

    def _extractDailyWeather(self, x, feature_dict):
        daily_weather = self.weather_data.getWeather(x['start_datetime'])
        feature_dict['DailyRainfall'] = self._getDailyRainfallValue(daily_weather['PRCP'])

    def _getDailyRainfallValue(self, rainfall):
        if rainfall == 0:
            return 'No_rainfall'
        elif rainfall < 100:
            return 'Less_than_1_inch'
        else:
            return 'Greater_than_1_inch'

    def _extractHourlyWeather(self, x, feature_dict):
        hourly_weather = self.weather_data.getHourlyWeather(x['start_datetime'])
        feature_dict['HourlyRainfall'] = self._getHourlyRainfallValue(hourly_weather['PRCP'], True)

    def _getHourlyRainfallValue(self, rainfall, use_buckets=True):
        if use_buckets:
            if rainfall == 0:
                return 'No_rainfall'
            elif rainfall < 10:
                return 'Less_than_0.1_inches'
            else:
                return 'Greater_than_0.1_inches'
        else:
            return rainfall

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

    def getFeatureDict(self, x):
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
        if self.config.getboolean(FEATURE_SELECTION, 'Zone_HourOfDay'):
            self._extractZoneHourOfDay(x, feature_dict)
        if self.config.getboolean(FEATURE_SELECTION, 'DailyWeather'):
            self._extractDailyWeather(x, feature_dict)
        if self.config.getboolean(FEATURE_SELECTION, 'HourlyWeather'):
            self._extractHourlyWeather(x, feature_dict)
        if self.config.getboolean(FEATURE_SELECTION, 'Zone_DayOfWeek_Hour'):
            self._extractZoneDayHour(x, feature_dict)
        if self.config.getboolean(FEATURE_SELECTION, 'Zone_DayOfWeek_HourlyWeather'):
            self._extractZoneDayOfWeekHourlyWeather(x, feature_dict)

        return feature_dict