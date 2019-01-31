
from utils import Utils
from Data_mgmt import Data_mgmt
from neural_model import Neural_Model
import os

data  = Data_mgmt()
utils = Utils()

dataset = data.read_dataset()

data.encode_data(dataset)
# data.stats_for_station()

data.iterate()
data.scale_dataset()

dato = data.supervised_learning()

data.split_sets(0.6, 0.38, 0.02)

m = Neural_Model()
m.fit_model()

os.system("clear; mdv README.md")
