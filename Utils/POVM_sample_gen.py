import numpy as np
from qutip import *
from itertools import product
from numpy import sqrt
from random import choices
from tqdm import tqdm




X = sigmax()
Y =  sigmay()
Z = sigmaz()
I = qeye(2)

def random_local_transform(n):
    U = rand_unitary_haar(2)
    return tensor([U]*n)


def MCMC_tetra_sampler(psi, n, n_samples, burnin=5000  ):
    X = sigmax()
    Y =  sigmay()
    Z = sigmaz()
    I = qeye(2)
    r = []
    r.append( 0.25 * (  I +  Z    )   )
    r.append(0.25 * (  I + (2* sqrt(2) / 3) * X  - (1.0/3.0) * Z  ))
    r.append(0.25 * (  I - ( sqrt(2) / 3) * X    + sqrt (2.0/3.0) * Y     - (1.0/3.0) * Z  ) )
    r.append (0.25 * (  I - ( sqrt(2) / 3) * X    - sqrt (2.0/3.0) * Y     - (1.0/3.0) * Z  ))
    state = np.ones(n, dtype=np.int16)


    samples_dict = {}
    def conditional(psi, u , state, r):
        s = state[:]
        p = np.zeros(4)
        for a in range(4):
            s[u] =  a
            E = tensor([r[k] for k in s ] )
            p[a] = expect(E , psi)
        return p

    def conditional2(psi, u , state, r, n):
        if 0 < u < n-1:
            E1 = tensor([r[k] for k in state[:u] ] )
            E2 =  tensor([r[k] for k in state[u+1:] ] )
            p = [ expect(tensor([E1, r[a], E2]) , psi) for a in range(4)]
        elif u ==0:
            E2 =  tensor([r[k] for k in state[u+1:] ] )
            p = [ expect(tensor([r[a], E2]) , psi) for a in range(4)]
        elif u == n-1:
            E1 = tensor([r[k] for k in state[:u] ] )
            p = [ expect(tensor([E1, r[a]]) , psi) for a in range(4)]
        return p

    def conditional3(psi, u , E1, E2, r):
        if 0 < u < n-1:

            p = [ expect(tensor([E1, r[a], E2]) , psi) for a in range(4)]
        elif u ==0:

            p = [ expect(tensor([r[a], E2]) , psi) for a in range(4)]
        elif u == n-1:

            p = [ expect(tensor([E1, r[a]]) , psi) for a in range(4)]
        return p

    state_list = []



    p = np.zeros(4)
    for t in tqdm(range(burnin)):
        E1 =  Qobj(1.0)
        #print(state)
        for u in range(n):
            if u < n-1:
                E2 =  tensor([r[k] for k in state[u+1:] ] )

            #p =  conditional2(psi, u, state, r,n)
            p = conditional3(psi, u , E1, E2, r)
            state[u] = choices(population=[0,1,2,3], weights = p)[0]
            if u == 0:
                E1 =  r[state[u]]
            else:
                E1 = tensor([E1,r[state[u]] ])



    for t in tqdm(range(n_samples)):
        E1 =  Qobj(1.0)
        for u in range(n):
            if u < n-1:
                E2 =  tensor([r[k] for k in state[u+1:] ] )

            #p =  conditional2(psi, u, state, r,n)
            p = conditional3(psi, u , E1, E2, r)
            state[u] = choices(population=[0,1,2,3], weights = p)[0]
            if u == 0:
                E1 =  r[state[u]]
            else:
                E1 = tensor([E1,r[state[u]] ])
        #print(state)

        state_list.append(list(state[:]))
        #if tuple(state) in samples_dict:
        #    samples_dict[tuple(state)] += 1
        #else:
        #    samples_dict[tuple(state)] = 1

    #b = [list(j) for j,w in samples_dict.items()]
    #B =  np.vstack(b)
    #A =  np.zeros((len(b), n+1 ))
    #A[:, 1:] =  B
    #W =  np.asarray( [w for j,w in samples_dict.items()])
    #A[:, 0 ] = W

    return state_list

def GHZ (n):
    #Create the n qubit GHZ state
    N =  np.power (2, n)

    psi = (basis(N,0) +  basis (N, N-1)).unit()
    psi.dims =  [[2]*n, [1]*n]
    return psi

def W_state(n, coeff ):
    #Create the n qubit W state

    N =  np.power(2 ,n)
    psi = Qobj()

    for i in range(n):
        psi += coeff[i]* basis(N, np.power(2,i) )

    psi.dims =  [[2]*n, [1]*n]
    return psi.unit()



def Tetra_POVM (n):
    #Generate the tetrahedral POVM operators for n qubits

    X = sigmax()
    Y =  sigmay()
    Z = sigmaz()
    I = qeye(2)
    r = []
    r.append( 0.25 * (  I +  Z    )   )
    r.append(0.25 * (  I + (2* sqrt(2) / 3) * X  - (1.0/3.0) * Z  ))
    r.append(0.25 * (  I - ( sqrt(2) / 3) * X    + sqrt (2.0/3.0) * Y     - (1.0/3.0) * Z  ) )
    r.append (0.25 * (  I - ( sqrt(2) / 3) * X    - sqrt (2.0/3.0) * Y     - (1.0/3.0) * Z  ))


    POVM = {}
    for K in product(range(4),  repeat = n):
        op_list = []
        for j in range(n):
            op_list.append( r[K[j]] )
            P = tensor (op_list)
            POVM[K] = P

    return POVM


def Pauli_POVM (n):
    #Generate the tetrahedral POVM operators for n qubits
    X = sigmax()
    Y =  sigmay()
    Z = sigmaz()
    I = qeye(2)
    r = []
    r.append( (1/6) * (  I +  Z    )  )
    r.append( (1/6) * (  I -  Z    )  )
    r.append( (1/6) * (  I +  X    )  )
    r.append( (1/6) * (  I -  X    )  )
    r.append( (1/6) * (  I +  Y    )  )
    r.append( (1/6) * (  I -  Y    )  )


    POVM = {}
    for K in product(range(6),  repeat = n):
        op_list = []
        for j in range(n):
            op_list.append( r[K[j]] )
            P = tensor (op_list)
            POVM[K] = P

    return POVM



def POVM_dist (psi,  n, POVM ):
    P = POVM
    prob_dist = {}
    for K in P.keys():
        prob_dist[K] = expect (P[K], psi)
    return prob_dist




def GHZ_samples (n, n_samples, POVM_type="Tetra", write=False):


    psi = GHZ (n)
    if POVM_type == "Pauli":
        P =  Pauli_POVM(n)
        P_dist = POVM_dist ( psi, n, P)

    if POVM_type == "Tetra":
        P =  Tetra_POVM(n)
        P_dist = POVM_dist ( psi, n, P)
    state_list=  choices ( population =  list ( P_dist.keys() ) , weights= P_dist.values() , k = n_samples )
    if write:
        np.savetxt("GHZ_samples.csv", state_list,fmt='%d')
    return state_list


def W_samples (n, n_samples,POVM_type="Tetra", write=True ):

    psi = W_state (n)
    if POVM_type == "Pauli":
        P =  Pauli_POVM(n)
        P_dist = POVM_dist ( psi, n, P)

    if POVM_type == "Tetra":
        P =  Tetra_POVM(n)
        P_dist = POVM_dist ( psi, n, P)
    state_list=  choices ( population =  list ( P_dist.keys() ) , weights= P_dist.values() , k = n_samples )
    if write:
        np.savetxt("W_samples.csv", state_list,fmt='%d')
    return state_list

def Haar_samples (n, n_samples, POVM_type="Tetra", write=True ):

    psi = rand_ket_haar (np.power(2, n ))
    psi.dims =  [[2]*n, [1]*n]

    if POVM_type == "Pauli":
        P =  Pauli_POVM(n)
        P_dist = POVM_dist ( psi, n, P)


    if POVM_type == "Tetra":
        P =  Tetra_POVM(n)
        P_dist = POVM_dist ( psi, n, P)
    state_list=  choices ( population =  list ( P_dist.keys() ) , weights= P_dist.values() , k = n_samples )
    if write:
        np.savetxt("Haar_samples.csv", state_list,fmt='%d')
    return state_list



def state_samples (psi, n, n_samples , POVM_type = "Tetra", write = True):

    psi.dims =  [[2]*n, [1]*n]

    if POVM_type == "Pauli":
        P =  Pauli_POVM(n)
        P_dist = POVM_dist ( psi, n, P)


    if POVM_type == "Tetra":
        P =  Tetra_POVM(n)
        P_dist = POVM_dist ( psi, n, P)
    state_list=  choices ( population =  list ( P_dist.keys() ) , weights= P_dist.values() , k = n_samples )
    if write:
        np.savetxt("samples.csv", state_list,fmt='%d')
    return state_list


def Hamiltonian(N, h, Jx, Jy, Jz):

    si = qeye(2)
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()

    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = sx
        sx_list.append(tensor(op_list))

        op_list[n] = sy
        sy_list.append(tensor(op_list))

        op_list[n] = sz
        sz_list.append(tensor(op_list))




    # construct the hamiltonian
    H = 0

    # energy splitting terms
    for n in range(N):
        H +=  h[n] * sx_list[n]

    # interaction terms
    for n in range(N-1):
        H +=  Jx[n] * sx_list[n] * sx_list[n+1]
        H +=  Jy[n] * sy_list[n] * sy_list[n+1]
        H +=  Jz[n] * sz_list[n] * sz_list[n+1]

     #Add periodic boundary conditions
    H +=  Jx[N-1] * sx_list[N-1] * sx_list[0]
    H +=  Jy[N-1] * sy_list[N-1] * sy_list[0]
    H +=  Jz[N-1] * sz_list[N-1] * sz_list[0]




    return H
