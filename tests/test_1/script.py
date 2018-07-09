from functions import *
from statistics import *


readWeatherData("H")

datosLeidos = readFile("IRALA")

print(datosLeidos)

average = averageBikes(datosLeidos)

# one_plot('Time of the day', 'Free bikes', datosLeidos, 'usage', 10, "Zunzunegi")

values, scaler = formatData(datosLeidos)

train_x, train_y, test_x, test_y  = split_dataset(values)

model = create_model(train_x)

train_model(model, train_x, train_y, test_x, test_y)

evaluate_model(scaler, model, test_x, test_y)

# predict(model, scaler, average)