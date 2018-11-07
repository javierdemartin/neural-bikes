
from utils import Utils
from Data_mgmt import Data_mgmt
from neural_model import Neural_Model

data  = Data_mgmt()
utils = Utils()

dataset = data.read_dataset()

data.encode_data(dataset)
data.stats_for_station()

data.iterate()

data.scale_dataset()

dato = data.supervised_learning()

data.split_sets(0.7, 0.28, 0.02)

m = Neural_Model(50, 300)
m.fit_model()	
