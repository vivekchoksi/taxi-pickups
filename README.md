Description
==============
Repository for our CS221 project for Autumn 2014.

Quick description
=================
If you wish to run our code right away, you can execute the following
to run the linear model on 1000 data points:

```bash
cd src
python taxi_pickups.py -m linear -n 1000 --features feature-sets/features1.cfg -v

# If you get a dependencies error, try running the following:
sudo pip install -r requirements.txt

# Or the following if you are on Stanford's corn machines:
pip install --user MySQL-python==1.2.5
pip install --user pybrain
```

#### Set up local MySQL database
```MySQL
mysql -u root
CREATE DATABASE taxi_pickups;
use taxi_pickups;


# Note: before running any sql script, make sure to edit the source to have
# file paths that are correct on your computer.

# To use our already aggregated data:
source /absolute/path/to/repo/sql/load_aggregated_from_csv.sql;

# To use the raw data downloaded from [Chris Whong's website](http://chriswhong.com/open-data/foil_nyc_taxi/):
source /absolute/path/to/repo/sql/load_raw_from_csv.sql;
source /absolute/path/to/repo/sql/aggregate_pickups.sql;
```

#### Download Python dependencies using [pip](https://pip.pypa.io/en/latest/)
```bash
sudo pip install -r requirements.txt
```

#### Run the program
```bash
cd src

# Sample usages...
# ... train and test a model.
python taxi_pickups.py --model linear --local --features feature-sets/features1.cfg --verbose

# ... plot a learning curve.
python plot_learning_curve.py --model linear --local --features feature-sets/features1.cfg --verbose

# ... create plots describing the data set (e.g. plot number of pickups by hour of day).
python plot.py --local

# ... submit a job to Stanford's SGE system on the barley machines.
python submit_job_barley.py <model-parameters>

# For more options and details about parameters:
python taxi_pickups.py --help
python plot_learning_curve.py --help
python plot.py --help
```


#### Set up remote MySQL database using AWS

First, set up an AWS instance.

```bash
# Copy local MySQL database table to AWS instance:
sudo mysqldump -u root --single-transaction --compress --order-by-primary taxi_pickups \
pickups_aggregated | mysql -h <instance name>.rds.amazonaws.com -P 3306 -u nyc -p taxi_pickups
```

# Connect to the remote MySQL instance from your machine:
mysql -h <instance name>.rds.amazonaws.com -P 3306 -u nyc -p taxi_pickups
```
