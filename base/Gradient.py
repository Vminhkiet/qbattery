import numpy as np
import matplotlib.pyplot as plt
from qiskit.primitives import Sampler
import qiskit
from qiskit import Aer, QuantumCircuit, transpile, assemble
import base.constant
import base.hamiltonian
import base.object
import tqix as tq
def Pmax(num_qubits, t1,thetas,shots):
    """
        Return Pmax

        INPUT:
          t: time
          thetas: array parameter (length 2*(N-1))
          shots: number of measurement times

    """
    time=np.linspace(0,t1,shots)
    arrayW=[]
    arrayP=[]
    for t in time:
        w = base.object.E(num_qubits, t,thetas) - base.object.E(num_qubits, 0,thetas)
        if(t!=0):
            p = w/t
        else:
            p = 0
        arrayW.append(w)
        arrayP.append(p)
    max_value = np.max(arrayP)
    return max_value



def gradient(num_qubits, t1,thetas, index):

    pi=base.hamiltonian.pi(index,num_qubits)
    pst=base.object.psi_t(num_qubits,t1,thetas)
    h0=base.hamiltonian.h0(num_qubits)
    return np.real(((1j*t1*tq.daggx(pst)@(pi@h0-h0@pi)@pst))[0,0])


def gradient_descent(num_qubits, t1,thetas,iteration):

    initial_params = thetas
    learning_rate = base.constant.learning_rate
    max_iterations = iteration
    thetass=[]
    arrayP=[]
    thetass.append(initial_params)
    maxP=Pmax(num_qubits, t1,initial_params,100)
    arrayP.append(maxP)
    for _ in range(iteration):
        gradients = [gradient(num_qubits,t1,initial_params, i) for i in range(len(initial_params))]
        initial_params = initial_params - learning_rate * np.array(gradients)
        arrayP.append(Pmax(num_qubits,t1,initial_params,1024))
        thetass.append(initial_params)
    max_value=np.max(arrayP)
    return max_value


