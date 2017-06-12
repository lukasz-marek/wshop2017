# -*- coding: utf-8 -*-
"""
Created on Sun May 21 21:56:35 2017

@author: Łukasz Marek
"""
from random import choice, uniform
from keras.layers.recurrent import GRU
from keras.models import Sequential 
from keras.utils import np_utils
from keras.layers import Dense
import numpy as np
import pandas as pd
import h5py

class DataTransformer:
    _pad = "?"
    _start = "^"
    _end = "$"
    
    def __init__(self, X, Y, sequence_length):
        characters = set([x[0].upper() for x in X.tolist() + Y.tolist()])
        characters.update(set([x[0].lower() for x in X.tolist() + Y.tolist()]))
        characters.add(DataTransformer._pad)
        self._decoder = dict(enumerate(characters))
        self._encoder = {v:k for k,v in self._decoder.items()}
        self._sequence_length = sequence_length
        self.X = self._encode_X(X)
        self.Y = self._encode_Y(Y)
        self._std = self.X.std()
        self._mean = self.X.mean()
        self.X = self._standardise(self.X)
        self.X = np.reshape(self.X, (len(self.X), self._sequence_length, 1))
    
    def _reshape(self, x):
        return np.reshape(x, (1, len(x), 1))
    
    def _standardise(self,X):
        return (X - self._mean) / self._std
        
    def _encode_X(self,X):
        X_Training = []
        for x in X:
            encoded = list(map(lambda letter: self._encoder[letter],x))
            while len(encoded) < self._sequence_length:
                encoded.append(self._encoder[DataTransformer._pad])
            X_Training.append(encoded[::-1])
        return np.array(X_Training)
    
    def _encode_Y(self, Y):
        Y_Training = []
        for y in Y:
            Y_Training.append(self._encoder[y])
        return np_utils.to_categorical(np.array(Y_Training))
    
    def encode(self, text):
        to_encode = text
        if len(to_encode) < self._sequence_length:
            to_encode += DataTransformer._pad * (self._sequence_length - len(to_encode))
        encoded = list(map(lambda letter: self._encoder[letter],to_encode))
        return self._standardise(np.array(encoded[::-1]))
    
    def decode(self,array):
        decoded = array * self._std + self._mean
        decoded = filter(lambda x: x!= DataTransformer._pad,map(lambda x: self._decoder[round(x)],reversed(decoded.tolist())))
        return "".join(decoded)
    
    def generate(self, model, limit = None):
        def generate_randomizing_vector():
            seq = []
            for _ in range(self.Y.shape[1]):
                seq.append(uniform(0,1))
            return np.array(seq)
        starter = choice(list(set(self._decoder.values()) 
        - set([DataTransformer._start, DataTransformer._end, DataTransformer._pad," "]))).upper()
        pattern = self.encode(DataTransformer._start + starter)
        pattern_as_list = [DataTransformer._start, starter]
        prediction = self._decoder[np.argmax(model.predict(self._reshape(pattern), verbose=0) + generate_randomizing_vector())]
        pattern_as_list.append(prediction)
        generated = DataTransformer._start + starter + prediction
        count = limit
        while prediction != DataTransformer._end:
            if len(pattern_as_list) > self._sequence_length:
                pattern_as_list.pop(0)
            pattern = self._reshape(self.encode("".join(pattern_as_list)))
            prediction = self._decoder[np.argmax(model.predict(pattern, verbose=0))]
            pattern_as_list.append(prediction)
            generated += prediction
            if count != None:
                count -= 1
                if count == 0:
                    break
        return generated

SEQUENCE_LENGTH = 4
       
data = pd.read_csv("training_data.csv", sep = ";",header = None)
X = data.get_values()[:,0]
Y = data.get_values()[:,-1]

transformer = DataTransformer(X,Y,SEQUENCE_LENGTH)


LAYER_SIZE = 30
NUMBER_OF_HIDDEN_LAYERS = 3
    
model = Sequential()
X_train = transformer.X[:int(len(transformer.X) * 0.8)]
X_test = transformer.X[int(len(transformer.X) * 0.8):]
Y_train = transformer.Y[:int(len(transformer.Y) * 0.8)]
Y_test = transformer.Y[int(len(transformer.Y) * 0.8):]

model.add(GRU(LAYER_SIZE, input_shape=(transformer.X.shape[1],
                                                     transformer.X.shape[2]),return_sequences = True))
for _ in range(NUMBER_OF_HIDDEN_LAYERS -1):
    model.add(GRU(LAYER_SIZE, return_sequences=True))
model.add(GRU(LAYER_SIZE))
model.add(Dense(transformer.Y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
for i in range(50):
    model.fit(X_train, Y_train, epochs=1, batch_size=180000, verbose=True, validation_data=(X_test,Y_test))
    model.save("generator.h5")