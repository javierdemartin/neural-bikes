
from utils import Utils
from Data_mgmt import Data_mgmt
from neural_model import Neural_Model
import os

data  = Data_mgmt()
utils = Utils()
model = Neural_Model()

# dataset = data.read_dataset()
# data.iterate()
# data.scale_dataset()
# dato = data.supervised_learning()

data.prepare_tomorrow()
model.tomorrow()