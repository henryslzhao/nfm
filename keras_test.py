
from __future__ import print_function
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout, Reshape, TimeDistributedDense,RepeatVector,Merge
from keras.layers.recurrent import GRU,LSTM
from keras.datasets.data_utils import get_file
import numpy as np
import re

from collections import Counter
import sys
from keras.models import model_from_json
from keras import backend as K 
from keras.engine.topology import Layer 

'''
this is for test BiasedMF
r = b_u + b_i + <u, i>
[koren 2009]
'''

u  = [0, 1, 2, 2, 3, 4, 4]
i   = [0, 1, 2, 2, 3, 4, 4]
z  = [1, 2, 0, 1, 1, 0, 1]
r = [1, 2, 4, 3, 3, 2, 1]
u = np.array(u)
i = np.array(i)
z = np.array(z)
r = np.array(r)

vec_dim = 200
bias_dim = 1
#rr = np.random.random((len(u), vec_dim))
ue = u
ie = i 
num_uid = 5
num_item = 5
num_t = 3



uModel = Sequential()
uModel.add(Embedding(num_uid, output_dim=vec_dim, input_length=1))
uModel.add(Reshape((vec_dim,)))



lModel = Sequential()
lModel.add(Embedding(num_uid, output_dim=vec_dim, input_length=1))
lModel.add(Reshape((vec_dim,)))


fModel = Sequential()
fModel.add(Merge([uModel, lModel], mode='dot', dot_axes=1))
fModel.add(Reshape((1,)))

ubModel = Sequential()
ubModel.add(Embedding(num_uid, output_dim=bias_dim, input_length=1))
ubModel.add(Reshape((bias_dim,)))


lbModel = Sequential()
lbModel.add(Embedding(num_uid, output_dim=bias_dim, input_length=1))
lbModel.add(Reshape((bias_dim,)))


model = Sequential()
model.add(Merge([fModel, ubModel, lbModel], mode='concat'))
model.add(Dense(1, ))                 
                 
model.compile(optimizer='sgd', loss='mse')
#print (fModel.get_config())
model.fit([u, i, u, i], r,nb_epoch=300)
p = model.predict([ue,ie, ue, ie])
print(p)


