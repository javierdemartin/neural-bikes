#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Data_mgmt import Data_mgmt
from neural_model import Neural_Model
import os

data  = Data_mgmt()
model = Neural_Model()

data.prepare_tomorrow()
data.prepare_today()

model.tomorrow(append_to_db = True)
