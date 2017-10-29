###############################
### Javier de Martin - 2017 ###
###############################

import xml.etree.ElementTree as ET
import urllib2
import string
import time
import os
import datetime
import MySQLdb
import collections
import codecs

# URL containing the XML feed
url = "https://tfl.gov.uk/tfl/syndication/feeds/cycle-hire/livecyclehireupdates.xml"

# Get current weekday
weekno = -1
weekday = ""
weekdays = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
weekno = datetime.datetime.today().weekday()
weekday = weekdays[weekno]


user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
headers={'User-Agent':user_agent,} 

request = urllib2.Request(url, headers = headers)
u = urllib2.urlopen(request)

tree = ET.parse(u)
root = tree.getroot()

id          = ""
stationName = ""
freeBikes   = ""
freeDocks   = ""
totalQuery  = ""

for child in root:
    for x in child:
      
        station_aux = []        
        
        query = ""
 
        if x.tag == "name":        	
            stationName = x.text
        elif x.tag == "nbBikes":
            freeBikes = x.text
        elif x.tag == "nbEmptyDocks":
            freeDocks = x.text
        elif x.tag == "id":
            id = x.text
				
        query = time.strftime("%Y/%m/%d %H:%M") + "," + weekday + "," + id + "," + stationName + "," + freeBikes + "," + freeDocks + "\n" 

    totalQuery += query
	        
#print totalQuery

with codecs.open("/home/aholab/javier/tfg/tools/parsers/data/Londres.txt", "a", "utf8") as file:
	file.write(totalQuery)


# Upload local backup to Dropbox
#os.system("./dropbox_uploader.sh upload TFG/tools/parsers/data/Londres.txt /TFG/")
