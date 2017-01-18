
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


class BiasedMF:
    '''
    this is for test BiasedMF
    r = b_u + b_i + <u, i>
    [koren 2009]
    '''
    
    def __init__(self, num_user, num_item, vec_dim, nb_epoch=100, bias_dim=1):
        #embedding for user 
        uModel = Sequential()
        uModel.add(Embedding(num_uid, output_dim=vec_dim, input_length=1))
        uModel.add(Reshape((vec_dim,)))

        # embedding for item
        lModel = Sequential()
        lModel.add(Embedding(num_uid, output_dim=vec_dim, input_length=1))
        lModel.add(Reshape((vec_dim,)))

        #compute the dot product of latent features <u, i>
        fModel = Sequential()
        fModel.add(Merge([uModel, lModel], mode='dot', dot_axes=1))
        fModel.add(Reshape((1,)))
    
        #set the user bias as an embedding with 1 dimension
        ubModel = Sequential()
        ubModel.add(Embedding(num_uid, output_dim=bias_dim, input_length=1))
        ubModel.add(Reshape((bias_dim,)))

        #item bias
        lbModel = Sequential()
        lbModel.add(Embedding(num_uid, output_dim=bias_dim, input_length=1))
        lbModel.add(Reshape((bias_dim,)))

        #compute the output r = b_u + b_i + <u, i>
        model = Sequential()
        model.add(Merge([fModel, ubModel, lbModel], mode='concat'))
        #we use a dense layer to output final rating, r = b_g + w1*b_u + w2*b_i + w3 * <u,i>
        # better than r = b_u + b_i + <u, i>, faster to converge
        model.add(Dense(1, ))                 
    
        #use mse as loss, sgd to train
        model.compile(optimizer='sgd', loss='mse')
        self.model = model
        self.num_item = num_item
        self.num_user = num_user
        self.vec_dim = vec_dim
        self.nb_epoch = nb_epoch
    
    def fit(self, u, i, r):
        print ('num_iter: ', self.nb_epoch)
        self.model.fit([u, i, u, i], r, nb_epoch=self.nb_epoch)
    
    def predict(self, u, i):
        return self.model.predict([u,i,u,i])


if __name__ == '__main__':

    u  = [0, 1, 2, 2, 3, 4, 4]
    i   = [0, 1, 2, 2, 3, 4, 4]
    r = [1, 2, 4, 3, 3, 2, 1]
    u = np.array(u)
    i = np.array(i)
    r = np.array(r)

    vec_dim = 200

    num_uid = 5
    num_item = 5
    mf = BiasedMF(num_uid, num_item, vec_dim)
    mf.fit(u, i, r)
    p = mf.predict(u, i)
    print (p)
