
from utils import Utils
from Data_mgmt import Data_mgmt
from neural_model import Neural_Model

data  = Data_mgmt()
utils = Utils()

# dataset = data.read_dataset()

# dataset, list_of_stations = data.encode_data(dataset)
# data.stats_for_station()

# data.iterate()

# data.scale_dataset()

dato = data.supervised_learning()

data.split_sets(0.8, 0.19, 0.01)

# m = Neural_Model(100, 1000)
# m.fit_model()	
