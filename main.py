
from utils import Utils
from Data_mgmt import Data_mgmt
from Neural_model import Neural_Model
import os

data  = Data_mgmt()
utils = Utils()

dataset = data.new_read_dataset(save_path = "/data/Bilbao.pkl")

data.encode_data(read_path = "/data/Bilbao.pkl", save_path = "/debug/encoded_data/")
# 
data.iterate()
data.scale_dataset()
# 
dato = data.supervised_learning()
# 
data.split_sets(0.98, 0.0, 0.02)

# m = Neural_Model()
# m.fit_model()

# os.system("clear")
