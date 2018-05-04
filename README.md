<img src="resources/neural-bikes_logo.png" width=35% align="right" />

# ML predictions for Bike sharing services

Trainign a ML model to predict availability for each station.

## Tests

Until the neural network is fully functional, I am working adding new tests I am performing on the [`tests`](https://github.com/javierdemartin/neural-bikes/tree/master/tests) folder.

## Where does the data come from?

Most cities have Open Data portals like the one in [Bilbao](https://www.bilbao.eus/opendata/es/inicio). You can get a lot of information from there but unluckily what you need to train a neural network is a historic of data not real time data. I kind of solved this by gathering the data by myself.

## What's the use of this?

I am an iOS developer of [Bicis](https://itunes.apple.com/es/app/bicis-bilbon-bizi/id1275889928?mt=8). It makes predictions of the bike availability for each day of the week, they are not as precise as they could be. My plan is to make those predictions using `neural-bikes` so they improve.
