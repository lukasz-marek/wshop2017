# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:50:19 2017

@author: Łukasz Marek
"""
from random import choice

import pandas as pd

begin_symbol = "^"
end_symbol = "$"

names = pd.read_csv("names.csv",header=None).get_values()
surnames = pd.read_csv("surnames.csv",header=None).get_values()

names = names.tolist()
surnames = surnames.tolist()

with open("data.csv","w+") as output:

    limit = 100000
    for _ in range(limit):
        name = choice(names)[0]
        surname = choice(surnames)[0]
        output.write(begin_symbol + name +" " + surname + end_symbol + "\n")