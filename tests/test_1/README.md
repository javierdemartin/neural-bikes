# test 1

## Data gathering

Every five minutes a 

## Steps

1. **Read dataset**
	1. Remove unwanted columns (station ID, latitude, longitude)
	2. Split datetime into day of the year and hour
2. **Encode data**
	1. LabelEncoder the hour, weekday and station name
	2. Save each station data as a NumPy array to debug/encoded_data/XXX.npy
3. **Stats**
	1. Save and plot the average availability for each station
4. **Fill holes**