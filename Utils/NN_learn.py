import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
#import keras
from tensorflow.keras import layers



from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects

def swish(x):
    return x * K.sigmoid(x)



get_custom_objects().update({'swish': Activation(swish)})
tf.keras.backend.set_floatx('float64')



def save_model_wts(H):
    N = len(H)
    path_list = []
    for u in range(N):
        modelpath = "saved_model/model"+ str(u+1) +".h5"
        path_list.append(modelpath)
        H[u].save_weights(modelpath)
    print("Weights saved around"+modelpath)
    return path_list






def train_NN_ising(samples,model, epochs, batch_size=200, eta =0.01 ,callbacks=[]):
    n_variables =  len(samples[0])
    n_samples =  len(samples)
    learnedH = []
    STEPS_PER_EPOCH =  n_samples/batch_size
    for i in range(n_variables):
        learnedH.append(tf.keras.models.clone_model(model))

    def GRISE_loss( H_u, s_u):
        return  tf.reduce_mean(tf.exp(-1*tf.multiply(s_u,H_u)))

    for u in range(n_variables):
        indices =  list(range(n_variables))
        del indices[u]
        s_u =  samples[:, u].reshape(n_samples,1)
        s_bar_u =  samples[:, indices]

        learnedH[u].compile(optimizer=tf.keras.optimizers.Adam(eta),
              loss=GRISE_loss)

        print("Learning variable:", u)
        learnedH[u].fit(s_bar_u, s_u, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    return learnedH



def train_NN(samples,q,model, epochs, batch_size=200,  eta = 0.01, callbacks=[], trans_inv = False, verbose=2):
    n_variables =  len(samples[0])
    b =  -1.0/q
    a =  1.0 + b
    n_samples =  len(samples)
    learnedH = []
    v = verbose
    STEPS_PER_EPOCH =  n_samples/batch_size
    print("Cloning models")


    def GRISE_loss( H_u, s_u):
        return  tf.reduce_mean(tf.exp(-1* tf.reduce_sum(tf.multiply(s_u,H_u), axis=1)))

    var_list =  [0] if trans_inv else range(n_variables)

    for i in var_list:
        learnedH.append(tf.keras.models.clone_model(model))

    if trans_inv:
        for i in range(1, n_variables):
            learnedH.append(None)
    print("Entering training loop")
    for u in var_list:
        indices =  list(range(n_variables))
        del indices[u]
        s_u =  samples[:, u].reshape(n_samples,1).astype(int)
        s_u =  tf.one_hot(s_u, depth=q, on_value=a, off_value=b)[:,0,:]
        s_u =  tf.cast(s_u, tf.float64)
        #s_bar_u =  samples[:, indices] + 1  #This is for Julia
        print("Pre processing done")

        learnedH[u].compile(optimizer=tf.keras.optimizers.Adam(eta),
              loss=GRISE_loss)

        print("Learning variable:", u)
        learnedH[u].fit(samples[:, indices] + 1, s_u, epochs=epochs, batch_size=batch_size, verbose=v,callbacks=callbacks)

    if trans_inv==True:
        for u in range(1,n_variables):
            learnedH[u] = learnedH[0]

    return learnedH

def train_NN_from_wts(samples_array, q, model, epochs, batch_size=200,  eta = 0.01, stopping_crit=False, trans_inv = False, verbose=0):
    #First column of samples aray are the weights, the next columns are the numbers

    B = samples_array[:, 1:]
    W =  samples_array[:,0]
    n_variables  =  B.shape[1]
    n_samples = int(sum(W))
    n_unique =  len(W)

    W =  W/sum(W)
    b =  -1.0/q
    a =  1.0 + b
    learnedH = []
    v = verbose
    for i in range(n_variables):
        learnedH.append(tf.keras.models.clone_model(model))

    def loss(model, s_u, s_bar_u, wts):
        return  tf.reduce_sum (tf.multiply(wts, tf.exp(-1* tf.reduce_sum(tf.multiply(s_u,model(s_bar_u)), axis=1))) )
    def grad(model, s_u, s_bar_u, wts):
        with tf.GradientTape() as tape:
            loss_value = loss(model, s_u, s_bar_u, wts)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)


    var_list =  [0] if trans_inv else range(n_variables)

    for u in var_list:
        indices =  list(range(n_variables))
        del indices[u]
        s_u =  B[:, u].reshape(n_unique,1).astype(int)
        s_u =  tf.one_hot(s_u, depth=q, on_value=a, off_value=b)[:,0,:]
        s_u =  tf.cast(s_u, tf.float64)
        s_bar_u =  B[:, indices] + 1
        optimizer =  tf.keras.optimizers.Adam(eta)

        train_loss_results = []

        #samples = tf.cast(samples, dtype=tf.float64)
        input_wts = []
        s_bar_u_train =  tf.data.Dataset.from_tensor_slices(s_bar_u)
        w_train =  tf.data.Dataset.from_tensor_slices(W)
        s_u_train = tf.data.Dataset.from_tensor_slices(s_u)
        #input_wts_model = [learnedH[u].trainable_variables[0].numpy().copy()]

        train_dataset = tf.data.Dataset.zip( (s_u_train, s_bar_u_train , w_train) )
        train_dataset =  train_dataset.batch(batch_size)

        #train_dataset =  zip(s_u_train, s_bar_u_train , w_train)
        #T = [x  for x in train_dataset]
        print("Learning variable:", u)
        for t in range(epochs):
            epoch_loss_avg =  tf.keras.metrics.Sum()

            for (x,y,z) in train_dataset:
                loss_value, grads = grad(learnedH[u],x,y,z)
                optimizer.apply_gradients(zip(grads, learnedH[u].trainable_variables))
                epoch_loss_avg.update_state(loss_value)

            train_loss_results.append(epoch_loss_avg.result())

            print("Epoch {:03d}: Loss: {:.3f}".format(t, epoch_loss_avg.result()))




        #learnedH[u].fit(samples[:, indices] + 1, s_u, epochs=epochs, batch_size=batch_size, verbose=v)

    if trans_inv==True:
        for u in range(1,n_variables):
            learnedH[u] = learnedH[0]

    return learnedH



def NN_MCMC(H, n, n_samples, burn_in = 10000):
    def toss(prob):
        if np.random.random() < prob:
            return 1.0
        else:
            return -1.0

    state_list = []
    state = np.ones((1, n), dtype=np.float64)
    prob_dict = {}
    indices = []
    for u in range(n):
        a = list(range(n))
        a.remove(u)
        indices.append(a[:])

    for t in range(burn_in):
        for u in range(n):
            if (u, tuple(state[:].flatten())) in  prob_dict:
                p = prob_dict[(u, tuple(state[:].flatten()))]

            else:
                p = np.exp(H[u]( state[:, indices[u]] ))/ (2 * np.cosh(H[u]( state[:, indices[u]] )) )
                prob_dict[(u, tuple(state[:].flatten()))] = p

            state[0,u] = toss(p[0][0])
    print("burn in complete")
    c =  0.1
    for t in range(n_samples):


        if t/n_samples > c:
            print(c*100, "% complete" )
            c += 0.1


        for u in range(n):
            if (u, tuple(state[:].flatten())) in  prob_dict:
                p = prob_dict[(u, tuple(state[:].flatten()))]

            else:
                p = np.exp(H[u]( state[:, indices[u]] ))/ (2 * np.cosh(H[u]( state[:, indices[u]] )) )
                prob_dict[(u, tuple(state[:].flatten()))] = p


            state[0,u] = toss(p[0][0])
        state_list.append(state[:].flatten())
    return np.vstack(state_list)


def list2dict(s):
    d= {}
    for j in s:
        if j in d:
            d[j] += 1
        else:
            d[j] = 1
    return d


def TVD (s1, s2, ns):
    s1_list = [tuple( s1[i,:].astype(int) ) for i in range(ns)]
    s2_list = [tuple( s2[i,:].astype(int) ) for i in range(ns)]
    d1 =  list2dict(s1_list)
    d2 = list2dict(s2_list)
    TV =0.0
    for i in d1.keys():
        if i in d2:
            TV += abs(d1[i] - d2[i])
        else:
            TV += abs(d1[i])

    for i in d2.keys():
        if i not in d1:
            TV += abs(d2[i])

    return TV/(2.0*ns)
