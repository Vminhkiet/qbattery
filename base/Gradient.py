import numpy as np
import matplotlib.pyplot as plt
from qiskit.primitives import Sampler
import qiskit
from qiskit import Aer, QuantumCircuit, transpile, assemble
import base.constant
import base.hamiltonian
import base.object
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


def cost_function(num_qubits, t1,thetas):
    """
      Calculating cost function values

      INPUT:

        t: time
        thetas: array parameter (length 2*(N-1))


    
    """
    return -np.abs(Pmax(num_qubits, t1,thetas,100))


def parameter_shift_gradient(num_qubits, t1,thetas, index):
    """
      Calculating derivatives using parameter-shift-rule

      INPUT:

        t: time
        thetas: array parameter (length 2*(N-1))
        index: the index under consideration of the array
      
        
    """
    params_plus = thetas.copy()
    params_minus = thetas.copy()


    params_plus[index] += base.constant.delta
    params_minus[index] -= base.constant.delta

    cost_Plus=cost_function(num_qubits, t1,params_plus)
    cost_Minus=cost_function(num_qubits, t1,params_minus)
    gradient = (cost_Plus - cost_Minus) / (2 * base.constant.delta)

    return gradient


def gradient_descent(num_qubits, t1,thetas):
    """
      Parameter optimization using parameter-shift-rule

      INPUT:

        t: time
        thetas: array parameter (length 2*(N-1))

      OUTPUT:

        Returns arraymax and parameter array list
    """
    initial_params = thetas
    learning_rate = base.constant.learning_rate
    max_iterations = base.constant.max_iterations
    thetass=[]
    thetass.append(initial_params)
    maxP=Pmax(num_qubits, t1,initial_params,100)
    while True:

        gradients = [parameter_shift_gradient(num_qubits,t1,initial_params, i) for i in range(len(initial_params))]
        initial_params = initial_params - learning_rate * np.array(gradients)
        p1=Pmax(num_qubits, t1,initial_params,100)
        if p1 < maxP :
            initial_params=thetass[-1]
            break
        maxP=p1
        thetass.append(initial_params)
    return initial_params,thetass


