<p align="center">
  <img width="250" height="" src="resources/neural-bikes_logo.png">
</p>

# ML predictions for Bike sharing services

Trainign a ML model to predict availability for each station.

## Tests

Until the neural network is fully functional, I am working adding new tests I am performing on the [`tests`](https://github.com/javierdemartin/neural-bikes/tree/master/tests) folder.

### Test 1

* First contact with machine learning so not much to talk about it

### Test 2

❌ Problems:

* Neural network learns the values and repeats them the following interval

### Test 3

The main purpose of this test is to predict a whole day of availability by just using some real samples and then feeding the nn with the generated values.

❌ Problems:

* Giving more timesteps for the inputs doesn't seem to solve the previous problem

Things to do:

* Reset the neural network status after an epoch finishes

## Where does the data come from?

Most cities have Open Data portals like the one in [Bilbao](https://www.bilbao.eus/opendata/es/inicio). You can get a lot of information from there but unluckily what you need to train a neural network is a historic of data not real time data. I kind of solved this by gathering the data by myself.


## What's the use of this?

I am an iOS developer of [Bicis](https://itunes.apple.com/es/app/bicis-bilbon-bizi/id1275889928?mt=8). It makes predictions of the bike availability for each day of the week, they are not as precise as they could be. My plan is to make those predictions using `neural-bikes` so they improve.
