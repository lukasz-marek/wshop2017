# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 22:21:05 2017

@author: Łukasz Marek
"""

import pandas as pd

data = pd.read_csv("data_with_turbulences.csv", sep = ";", header = None).get_values()
X = data[:,0].tolist()
Y = data[:,-1].tolist()

characters = set()
for chain in X:
    for letter in set(chain):
        characters.update([letter.lower(),letter.upper()])
for chain in Y:
    for letter in set(chain):
        characters.update([letter.lower(),letter.upper()])