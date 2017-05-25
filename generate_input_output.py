# -*- coding: utf-8 -*-
"""
Created on Tue May 23 19:51:22 2017

@author: Łukasz Marek
"""
import pandas as pd

def generate_pairs(element, max_input_length):
    pairs = []
    symbols = [ x for x in element]
    input = [symbols.pop(0)]
    while len(input) > 0 and len(symbols) > 0:
        x = "".join(input)
        input.append(symbols.pop(0))
        y = input[-1]
        pairs.append((x,y))
        if len(input) > max_input_length:
            input.pop(0)
    return pairs
    
data = pd.read_csv("data.csv",header=None).get_values().tolist()
with open("training_data.csv","w+") as output:
    for element in data:
        pairs = generate_pairs(element[0],7)
        for x,y in pairs:
            output.write(x+";" + y + "\n")
    
    
