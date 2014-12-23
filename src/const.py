class Const(object):

    # Name of MySQL database that houses taxi pickup data.
    DATABASE_NAME = 'taxi_pickups'

    # Name of aggregated pickups data table.
    AGGREGATED_PICKUPS = 'pickups_aggregated_manhattan'

    # Paths to CSV data files.
    DAILY_WEATHER_DATA = '../data/nyc_observed_weather_daily.csv'
    HOURLY_WEATHER_DATA = '../data/nyc_observed_weather_hourly.csv'

    # Path to directory into which to save plots.
    OUTFILE_DIRECTORY = "../outfiles/"

    # Numbers of data points for batched model training and evaluation.
    TRAIN_BATCH_SIZE = 40000
    TEST_BATCH_SIZE = 40000
