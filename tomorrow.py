#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils import Utils
from Data_mgmt import Data_mgmt
from neural_model import Neural_Model
import os

data  = Data_mgmt()
utils = Utils()
model = Neural_Model()

data.read_dataset(path = '/data/Bilbao.txt', save_path = '/data/Bilbao.pkl')

data.prepare_tomorrow()
model.tomorrow()
