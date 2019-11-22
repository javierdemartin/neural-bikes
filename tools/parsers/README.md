# Parsers

Web parsers used to get the availability data for each city. 

Written all of them in Python 2.7.

The collected data is under [`data`](https://github.com/javierdemartin/TFG/tree/master/tools/parsers/data) folder, updated frequently (every week or so) as a backup.

## Available Cities

List of all the cities analyzed with their respective feed URL, some of them provide the data in XML and others in JSON.

* [Barcelona](http://wservice.viabicing.cat/v1/getstations.php?v=1.xml) _Started 2017/09/29_ [Download](https://www.dropbox.com/s/u458i2yk5ydpjo6/Barcelona.txt?dl=1)
* [Bilbao](http://www.bilbao.eus/WebServicesBilbao/WSBilbao?s=ODPRESBICI&u=OPENDATA&p0=A&p1=A) _Started 2017/09/29_ [Download](https://www.dropbox.com/s/fsf4cxb5j3b17pt/Bilbao.txt?dl=1)
* [Goteborg](https://api.jcdecaux.com/vls/v1/stations?contract=Goteborg&apiKey=9fcde589b2071fa7895969c4f0a186f2beb6ac84) _Started 2017/10/02_ [Download](https://www.dropbox.com/s/yunm9clbbr1sf3d/Goteborg.txt?dl=1)
* [Ljubljana](https://api.jcdecaux.com/vls/v1/stations?contract=Ljubljana&apiKey=9fcde589b2071fa7895969c4f0a186f2beb6ac84) _Started 2017/10/02_ [Download](https://www.dropbox.com/s/3m8vausvzjmffll/Ljubljana.txt?dl=1)
* [London](https://tfl.gov.uk/tfl/syndication/feeds/cycle-hire/livecyclehireupdates.xml) _Started 2017/10/6_ [Download](https://www.dropbox.com/s/zsg7w5grn4qbctz/Londres.txt?dl=1)
* [Luxembourg](https://api.jcdecaux.com/vls/v1/stations?contract=Luxembourg&apiKey=9fcde589b2071fa7895969c4f0a186f2beb6ac84) _Started 2017/10/02_ [Download](https://www.dropbox.com/s/bitbn2wjg8q4lc6/Luxembourg.txt?dl=1)
* [Madrid](https://rbdata.emtmadrid.es:8443/BiciMad/get_stations/WEB.SERV.javierdemartin@me.com/4A3332DE-1A06-4C88-9B23-66C88B2A351A) _Started 2017/09/29_ [Download](https://www.dropbox.com/s/xoysbe45ytmea6x/Madrid.txt?dl=1)
* [Nantes](https://api.jcdecaux.com/vls/v1/stations?contract=Nantes&apiKey=9fcde589b2071fa7895969c4f0a186f2beb6ac84) _Started 2017/10/02_ [Download](https://www.dropbox.com/s/cs3ewwbdeld1qpw/Nantes.txt?dl=1)
* [Paris](https://api.jcdecaux.com/vls/v1/stations?contract=Paris&apiKey=9fcde589b2071fa7895969c4f0a186f2beb6ac84) _Started 2017/10/02_ [Download](https://www.dropbox.com/s/i1gj2a893l0nx1d/Paris.txt?dl=1)
* [Santander](https://api.jcdecaux.com/vls/v1/stations?contract=Santander&apiKey=9fcde589b2071fa7895969c4f0a186f2beb6ac84) _Started 2017/10/02_ [Download](https://www.dropbox.com/s/rk6eqebtnrku8b6/Santander.txt?dl=1)
* [Sevilla](https://api.jcdecaux.com/vls/v1/stations?contract=Seville&apiKey=9fcde589b2071fa7895969c4f0a186f2beb6ac84) _Started 2017/10/02_ [Download](https://www.dropbox.com/s/jjzadkeh1rzkh5k/Sevilla.txt?dl=1)
* [Toulouse](https://api.jcdecaux.com/vls/v1/stations?contract=Toulouse&apiKey=9fcde589b2071fa7895969c4f0a186f2beb6ac84) _Started 2017/10/02_ [Download](https://www.dropbox.com/s/33oj5zhm4tq2at3/Toulouse.txt?dl=1)

Goteborg, Ljublkana, Luxembourg, Nantes, Paris, Santander, Sevilla and Toulouse have the same XML structure. the data provider is the same adn the parsers are mostly the same.

## Problems and Solutions

In some cases I forgot to add a newline character at the end of a line, therefore all the data was written in the same line instead of in diferent ones.

To solve it I just had to add a newline character in the head of each line. Each line started with the year so the bash command I use was:

```bash
sed 's/2017/\n&/g' file.txt > file.txt
```
