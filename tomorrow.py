#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Data_mgmt import Data_mgmt
from cluster import Cluster

from neural_model import Neural_Model
import sys
import os

data  = Data_mgmt(city=sys.argv[1])
model = Neural_Model()
cluster = Cluster(city=sys.argv[1])
labels = cluster.do_cluster()

dataToPredict = data.prepare_tomorrow(labels)


print(dataToPredict)

model.tomorrow(data = dataToPredict, append_to_db = False)