#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Data_mgmt import Data_mgmt
import os

data  = Data_mgmt()

data.read_dataset(path = '/data/Bilbao.txt', save_path = '/data/Bilbao.pkl')

data.prepare_today()
