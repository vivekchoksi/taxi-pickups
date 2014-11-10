#!/usr/bin/python
import MySQLdb
import datetime

# connect
db = MySQLdb.connect(host="localhost", user="root", passwd="",
db="taxi_pickups")

cursor = db.cursor()



def getNumPickups(start_time, duration, start_latitude, start_longitude, side_length):
  # Format start and end time strings.
  datetime_format = "%Y-%m-%d %H:%M:%S"
  end_time = start_time + duration
  start_time_string = start_time.strftime(datetime_format)
  end_time_string = end_time.strftime(datetime_format)

  # Initialize end latitude and longitude variables.
  end_latitude = start_latitude + side_length
  end_longitude= start_longitude + side_length

  query_string = "SELECT COUNT(*) FROM trip_data WHERE " \
                    "`pickup_datetime` between '%s' AND '%s' AND " \
                    "`pickup_longitude` >= %f AND " \
                    "`pickup_longitude` < %f AND " \
                    "`pickup_latitude` >= %f AND " \
                    "`pickup_latitude` < %f" \
                  % (start_time_string, end_time_string, \
                     start_longitude, end_longitude, \
                     start_latitude, end_latitude)

  print "Querying `trip_data`: " + query_string
  cursor.execute(query_string_trivial)
  db.commit()
  return int(cursor.fetchone()[0])

# execute SQL select statement
start_time = datetime.datetime(2013, 1, 1, 1, 1, 1, 1)
num_pickups = getNumPickups(
  start_time=start_time,
  duration=datetime.timedelta(hours=1),
  start_latitude=40.75,
  start_longitude=-73.95,
  side_length=0.05,
)

print num_pickups