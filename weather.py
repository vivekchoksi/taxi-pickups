#!/usr/bin/python
from const import Const
import datetime

class Weather(object):
    '''
    Abstraction class for the weather data set.

    Data includes the following metrics for each day in 2013:
        * AWND - Average daily wind speed (tenths of meters per second)
        * SNOW - Snowfall (mm)
        * TMIN - Minimum temperature (tenths of degrees C)
        * TMAX - Maximum temperature (tenths of degrees C)
        * SNWD - Snow depth (mm)
        * PRCP - Precipitation (tenths of mm)

    Example usage:
        >>> w = weather()
        >>> daily_weather = w.getWeather(datetime.datetime(2013, 1, 15)) # 2013 January 15
        >>> print daily_weather['SNOW']
        37
        >>> print daily_weather['TMIN']
        -24

    Data collected from the NYC Central Park Observation Belevedere Tower, located
    at 40.78 degrees north and -73.97 degrees west. Data source:
    http://www.ncdc.noaa.gov/cdo-web/datasets/GHCND/stations/GHCND:USW00094728/detail
    '''
    weather = None

    def __init__(self):
        # Load data set
        self.weather = {}
        f = open(Const.WEATHER_DATA)
        headers = f.readline().strip().split(',')
        lines = f.readlines()
        for line in lines:
            # Each line corresponds to weather data for one day.
            raw_values = line.strip().split(',')
            date = datetime.datetime(int(raw_values[0]), # YEAR
                                     int(raw_values[1]), # MONTH
                                     int(raw_values[2])) # DAY
            values = {}
            for index in xrange(3, len(headers)):
                values[headers[index]] = int(raw_values[index])
            self.weather[date] = values

    def getWeather(self, date):
        '''
        Returns a dictionary with the weather values for the specified date.

        :param date: datetime object with year, month, and day attributes
        :return: dictionary whose keys are the types of recorded weather data
         available for this day.
        '''
        return self.weather[datetime.datetime(date.year, date.month, date.day)]