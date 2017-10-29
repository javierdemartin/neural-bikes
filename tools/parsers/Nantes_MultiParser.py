# -*- coding: utf-8 -*-

###############################
### Javier de Martin - 2017 ###
###############################

import xml.etree.ElementTree as ET
import urllib2
import urllib
import string
import time
import os
import datetime
from HTMLParser import HTMLParser
import re
import collections
import codecs
import json

# URL containing the XML feed
url = "https://api.jcdecaux.com/vls/v1/stations?contract=Nantes&apiKey=9fcde589b2071fa7895969c4f0a186f2beb6ac84"

response = urllib.urlopen(url)
data = json.loads(response.read())

# print data

print type(data[0])

print "$$$$$$$$$$$$$$$$$$$$$$$$$"

print data[0]

# json_data = json.loads(open(data[0], "r").read())
# 
# json_data = json.loads(data[0])
# 
# print type(json_data)



# Get current weekday
weekno = -1
weekday = ""
weekdays = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
weekno = datetime.datetime.today().weekday()
weekday = weekdays[weekno]

id          = ""
stationName = ""
freeBikes   = ""
freeDocks   = ""
query       = ""
status      = ""
totalQuery  = ""


for i in data:

#     print i

#     print i["name"]

    id          = str(i["number"])
    stationName = i["name"]
    status      = i["status"]
    freeBikes   = str(i["available_bikes"])
    freeDocks   = str(i["available_bike_stands"])

    query = time.strftime("%Y/%m/%d %H:%M") + "," + weekday + "," + id + "," + stationName + "," + freeBikes + "," + freeDocks + "\n"
    
    if status == "OPEN":
        totalQuery += query
    
    print query
#print totalQuery

with codecs.open("/home/aholab/javier/tfg/tools/parsers/data/Nantes.txt", "a", "utf8") as file:
    file.write(totalQuery)
#
## Upload local backup to Dropbox
#os.system("./dropbox_uploader.sh upload TFG/tools/parsers/data/Nantes.txt /TFG/")

