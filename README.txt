Description
==============
Repository for our CS221 project for Autumn 2014.

MySQL setup:
==============
mysql -u root
CREATE DATABASE taxi_pickups;
use taxi_pickups;
source /path/to/the/sql/script/load-trip-data.sql

Python packages setup
======================
sudo pip install -r requirements.txt


Running the program
====================
Sample usage: python taxi_pickups.py -m baseline
Run python taxi_pickups.py --help for more options


