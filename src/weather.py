#!/usr/bin/python
import datetime
from const import Const

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
        # Load daily weather.
        self.daily_weather = {}
        f_daily = open(Const.DAILY_WEATHER_DATA)
        headers = f_daily.readline().strip().split(',')
        lines = f_daily.readlines()
        for line in lines:
            # Each line corresponds to weather data for one day.
            raw_values = line.strip().split(',')
            date = datetime.datetime(int(raw_values[0]), # YEAR
                                     int(raw_values[1]), # MONTH
                                     int(raw_values[2])) # DAY
            values = {}
            for index in xrange(3, len(headers)):
                values[headers[index]] = int(raw_values[index])
            self.daily_weather[date] = values
        f_daily.close()

        # Load hourly weather.
        self.hourly_weather = {}
        f_hourly = open(Const.HOURLY_WEATHER_DATA)
        f_hourly.readline() # First line contains headers.
        lines = f_hourly.readlines()
        for line in lines:
            # Each line corresponds to weather data for one hour.
            raw_values = line.strip().split(',')
            date = datetime.datetime(int(raw_values[0]), # YEAR
                                     int(raw_values[1]), # MONTH
                                     int(raw_values[2]), # DAY
                                     int(raw_values[3])) # HOUR
            values = {'PRCP': int(raw_values[4])}  # Hundredths of an inch of precipitation.
            self.hourly_weather[date] = values
        f_hourly.close()

    def getDailyWeather(self, date):
        '''
        Returns a dictionary with the weather values for the specified date.

        :param date: datetime object with year, month, and day attributes
        :return: dictionary whose keys are the types of recorded weather data
         available for this day.
        '''
        return self.daily_weather[datetime.datetime(date.year, date.month, date.day)]

    def getHourlyWeather(self, date):
        '''
        Returns a dictionary with the weather values for the specified date and hour.

        :param date: datetime object with year, month, day, and hour attributes
        :return: dictionary whose keys are the types of recorded weather data
         available for this day.
        '''
        date_hour = datetime.datetime(date.year, date.month, date.day, date.hour)
        if date_hour in self.hourly_weather:
            hourly_weather = self.hourly_weather[datetime.datetime(date.year, date.month, date.day, date.hour)]
        else:
            hourly_weather = {'PRCP': 0}
        return hourly_weather
