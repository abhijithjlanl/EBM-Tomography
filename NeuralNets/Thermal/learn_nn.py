import numpy as np
#from qutip import *
import sys
sys.path.append("../../Utils")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#import networkx as nx
#from matplotlib import pyplot as plt
import tensorflow as tf
from NN_learn import *
#from POVM_sample_gen import *
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import layers

def swish(x):
    return x * K.sigmoid(x)

get_custom_objects().update({'swish': Activation(swish)})


#samples =  np.loadtxt("saved_model/samples_"+str(d)+"_"+str(n_samples)+".csv")
#samples =  np.loadtxt("saved_model/samples.csv")
n_samples =  int(sys.argv[1])
n =  int(sys.argv[2])
samples =  np.loadtxt(f"samples.csv")
print(samples.shape)
#samples =  samples.reshape(40000, 10)
batch_size =  5000
samples =  np.transpose(samples)
inds =  np.random.choice(samples.shape[1], n_samples, replace=False)
samples =  samples[:, inds] - 1
samples =  np.transpose(samples)
print(samples.shape)
assert n_samples == samples.shape[0], "looks like the samples have to be transposed"
q = 4
#samples =  samples[ :n_samples,:] - 1

n = samples.shape[1] 
schedule =  tf.keras.optimizers.schedules.InverseTimeDecay(
0.008 , decay_steps= 50*(n_samples//batch_size),
decay_rate=4,staircase=True)
n_variables = n
b =  -1.0/q
a =  1.0 + b
samples =  samples.astype('float32')
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, min_delta =1E-8, restore_best_weights=True)

path_list = []

def GRISE_loss( H_u, s_u):
        return  tf.reduce_mean(tf.exp(-1* tf.reduce_sum(tf.multiply(s_u,H_u), axis=1)))

var_list =  range(n_variables)

for u in var_list:
        indices =  list(range(n_variables))
        del indices[u]
        s_u =  samples[:, u].reshape(n_samples,1).astype(int)
        s_u =  tf.one_hot(s_u, depth=q, on_value=a, off_value=b)[:,0,:]
        s_u =  tf.cast(s_u, tf.float16)
        #s_bar_u =  samples[:, indices] + 1  #This is for Julia
        print("Pre processing done")
        width = 25
        model = tf.keras.Sequential()
        model.add(layers.Dense(width,activation=swish, input_shape=(n-1,)))
        model.add(layers.Dense(width, activation=swish))
        model.add(layers.Dense(width, activation=swish))
        model.add(layers.Dense(q))
      #  model.summary()
        
        
        

        model.compile(optimizer=tf.keras.optimizers.Adam(schedule),
              loss=GRISE_loss)
        print("Learning variable:", u)
        model.fit(samples[:, indices] + 1, s_u, epochs=500, batch_size= batch_size, verbose=0,callbacks=[callback])
        #print(GRISE_loss(model(samples[:, indices] + 1),  s_u))
        
        modelpath = "saved_model/model"+ str(u+1) +".h5"
        path_list.append(modelpath)
        model.save_weights(modelpath)



