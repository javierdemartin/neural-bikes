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
url = "https://rbdata.emtmadrid.es:8443/BiciMad/get_stations/WEB.SERV.javierdemartin@me.com/4A3332DE-1A06-4C88-9B23-66C88B2A351A"

response = urllib.urlopen(url)
data = json.loads(response.read())

json_data = json.loads(data["data"])

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


for i in json_data["stations"]:

    id = str(i["id"])
    stationName = i["name"]
    freeBikes = str(i["dock_bikes"] - i["reservations_count"])
    freeDocks = str(i["free_bases"])

    query = time.strftime("%Y/%m/%d %H:%M") + "," + weekday + "," + id + "," + stationName + "," + freeBikes + "," + freeDocks + "\n"
    totalQuery += query

#print totalQuery

with codecs.open("/home/aholab/javier/tfg/tools/parsers/data/Madrid.txt", "a", "utf8") as file:
    file.write(totalQuery)
#
## Upload local backup to Dropbox
#os.system("./dropbox_uploader.sh upload TFG/tools/parsers/data/Madrid.txt /TFG/")
#
