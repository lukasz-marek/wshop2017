# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:50:19 2017

@author: Łukasz Marek
"""
from random import choice, randrange
from string import ascii_letters
import pandas as pd

begin_symbol = "^"
end_symbol = "$"

names = pd.read_csv("names.csv",header=None).get_values()
surnames = pd.read_csv("surnames.csv",header=None).get_values()

names = names.tolist()
surnames = surnames.tolist()

def swap_two_random_characters(input):
    index_1 = randrange(len(input))
    index_2 = index_1 - 1 if index_1 == len(input) - 1 else choice([index_1 + 1, index_1 -1]) if index_1 > 0 else index_1 + 1
    letters = list(input)
    letters[index_1],letters[index_2] = letters[index_2],letters[index_1]
    return "".join(letters)

def switch_case(input):
    index = randrange(len(input))
    letters = list(input)
    letters[index] = letters[index].upper() if letters[index].islower() else letters[index].lower()
    return "".join(letters)

def delete_letter(input):
    index = randrange(len(input))
    letters = list(input)
    del letters[index]
    return "".join(letters)

def insert_random_letter(input):
    index = randrange(len(input))
    letters = list(input)
    letters.insert(index,choice(ascii_letters))
    return "".join(letters)

#Number of generated training samples    
TRAINING_DATA_SIZE = 5000

with open("data.csv","w+") as output:
    for _ in range(TRAINING_DATA_SIZE):
        name = choice(names)[0]
        surname = choice(surnames)[0]
        output.write(begin_symbol + name +" " + surname + end_symbol + "\n")
        
"""functions = [
        lambda x: x,
        lambda x: swap_two_random_characters(x),
        lambda x: switch_case(x),
        lambda x: delete_letter(x),
        lambda x: swap_two_random_characters(switch_case(x)),
        lambda x: swap_two_random_characters(delete_letter(x)),
        lambda x: switch_case(delete_letter(x)),    
        lambda x: insert_random_letter(x),
        lambda x: swap_two_random_characters(insert_random_letter(x)),
        lambda x: delete_letter(insert_random_letter(x))
        ]
        
with open("data_with_turbulences.csv","w+") as output:
    limit = 10000
    for _ in range(limit):
        name = choice(names)[0]
        surname = choice(surnames)[0]
        text = name + " " + surname
        #function = choice(functions)
        output.write(text + ";" + text + "\n")
        #output.write(function(text) + ";" + text + "\n")"""