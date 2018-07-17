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
import re, htmlentitydefs
from HTMLParser import HTMLParser
import re
import MySQLdb
import collections
import codecs

class colors:
    HEADER = '\033[95m'
    blue = '\033[94m'
    green = '\033[92m'
    yellow = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# URL containing the XML feed
url = "http://wservice.viabicing.cat/v1/getstations.php?v=1.xml"

h = HTMLParser()

# Get current weekday
weekno = -1
weekday = ""
weekdays = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
weekno = datetime.datetime.today().weekday()
weekday = weekdays[weekno]

request = urllib2.Request(url, headers={"Accept" : "application/xml"})
u = urllib2.urlopen(request)

tree = ET.parse(u)
root = tree.getroot()

id          = ""
stationName = ""
freeBikes   = ""
freeDocks   = ""
query       = ""
status      = ""
totalQuery  = ""

for child in root:

    parsed_station = {}

    for x in child:

        if x.tag == "street":

            aux = x.text
            aux = aux.replace("&#039;", r"'")

            aux = h.unescape(aux)
            
            stationName = aux

            parsed_station["street"] = aux
        elif x.tag == "id":
            parsed_station["id"] = x.text
            id = x.text
        elif x.tag == "status":
            status = x.text
        elif x.tag == "bikes":
			freeBikes = x.text
        elif x.tag == "slots":
			freeDocks = x.text
            
    query = time.strftime("%Y/%m/%d %H:%M") + "," + weekday + "," + id + "," + stationName + "," + freeBikes + "," + freeDocks + "\n"
    print query
            
    if status is not "CLS":
        if (",,,") not in query:
            totalQuery += query

with codecs.open("/home/aholab/javier/tfg/tools/parsers/data/Barcelona.txt", "a", "utf8") as file:
    file.write(totalQuery)

# Upload local backup to Dropbox
#os.system("./dropbox_uploader.sh upload TFG/tools/parsers/data/Barcelona.txt /TFG/")
