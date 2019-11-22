import numpy as np
from Plotter import Plotter
import os
from Data_mgmt import Data_mgmt
from influxdb import InfluxDBClient
import datetime
from datetime import timedelta
import pandas as pd


d = Data_mgmt()
d.plot_today()
