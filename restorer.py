# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 22:21:05 2017

@author: Łukasz Marek
"""

import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers.wrappers import TimeDistributed
from keras.layers import Activation
from keras.layers.core import RepeatVector


CHUNK_SIZE = 100

UNKNOWN = "<UNKNOWN>"
PADDING = "PADDING"
VOCABULARY_SIZE = 1000

HIDDEN_LAYER_SIZE = 1000
NUMBER_OF_HIDDEN_LAYERS = 2

MAX_SENTENCE_LENGTH = 100 

NUMBER_TO_LETTER = {0:PADDING}
NUMBER_TO_LETTER.update(dict({(i + 1, chr(i)) for i in range(VOCABULARY_SIZE)}))
NUMBER_TO_LETTER[VOCABULARY_SIZE + 1] = UNKNOWN
                
LETTER_TO_NUMBER = {PADDING:0}
LETTER_TO_NUMBER.update(dict({(chr(i), i + 1) for i in range(VOCABULARY_SIZE)}))
LETTER_TO_NUMBER[UNKNOWN] = VOCABULARY_SIZE + 1

def map_onto_numbers(sentence):
    return [LETTER_TO_NUMBER[x] if x in LETTER_TO_NUMBER else LETTER_TO_NUMBER[UNKNOWN] for x in sentence][0:MAX_SENTENCE_LENGTH]
                
def data():
    for chunk in pd.read_csv("data_with_turbulences.csv", sep=";", header=None, chunksize=CHUNK_SIZE):
        x = pad_sequences(list(map(map_onto_numbers, chunk.get_values()[:, 1].tolist())), 
                          maxlen=MAX_SENTENCE_LENGTH, dtype='int32')
        y = pad_sequences(list(map(map_onto_numbers, chunk.get_values()[:, -1].tolist())), 
                          maxlen=MAX_SENTENCE_LENGTH, dtype='int32')
        sequences = np.zeros((len(y), MAX_SENTENCE_LENGTH, len(LETTER_TO_NUMBER)))
        for i, sample in enumerate(y):
            for j, symbol in enumerate(sample):
                sequences[i, j, symbol] = 1
        yield x, sequences
    raise StopIteration

#encoder
model = Sequential()
model.add(Embedding(VOCABULARY_SIZE + 2, CHUNK_SIZE, input_length=MAX_SENTENCE_LENGTH, mask_zero=True))
print(model.output_shape)
model.add(LSTM(HIDDEN_LAYER_SIZE))
print(model.output_shape)
#decoder
model.add(RepeatVector(MAX_SENTENCE_LENGTH))
print(model.output_shape)
for _ in range(NUMBER_OF_HIDDEN_LAYERS):
    model.add(LSTM(HIDDEN_LAYER_SIZE, return_sequences=True))
print(model.output_shape)
model.add(TimeDistributed(Dense(VOCABULARY_SIZE + 2)))
print(model.output_shape)
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])
#training
for epoch in range(1):
    for X, Y in data():
        model.fit(X, Y, batch_size=20)