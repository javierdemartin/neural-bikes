
from utils import color, Utils, Data_mgmt, Neural_Model
import numpy as np

# utils.print_smth("HEH", 5)

data  = Data_mgmt()
utils = Utils()

# dataset = data.read_dataset()

# dataset, list_of_stations = data.encode_data(dataset)
# data.stats_for_station()

ret = data.iterate()

# scaled = data.scale_dataset(ret)

# dato = data.supervised_learning(scaled)

# train_x, train_y, validation_x, validation_y, test_x, test_y  = data.split_sets(dato, 0.8,0.19,0.01)

# m = Neural_Model(train_x, train_y, test_x, test_y, 100, validation_x, validation_y, 1000)
# m.fit_model()	
