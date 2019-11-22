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
url = "http://www.zaragoza.es/api/recurso/urbanismo-infraestructuras/estacion-bicicleta.xml?rf=html&results_only=false&srsname=wgs84"

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
totalQuery  = ""

for child in root:

    print child

    for detalle in child:
       
        station_aux = []        
        
        query = ""

        for x in detalle:
            
            if x.tag == "NOMBRE":        	
                id = x.text.split("-")[0]
                stationName = x.text.split("-")[1]
            elif x.tag == "ALIBRES":
                freeBikes = x.text
            elif x.tag == "BLIBRES":
                freeDocks = x.text
				
        query = time.strftime("%Y/%m/%d %H:%M") + "," + weekday + "," + id + "," + stationName + "," + freeBikes + "," + freeDocks + "\n"
        
        if (",,,") not in query:
	        totalQuery += query
	        
print totalQuery

with codecs.open("/home/aholab/javier/tfg/tools/parsers/data/Zaragoza.txt", "a", "utf8") as file:
	file.write(totalQuery)


# Upload local backup to Dropbox
#os.system("./dropbox_uploader.sh upload TFG/tools/parsers/data/Zaragoza.txt /TFG/")

