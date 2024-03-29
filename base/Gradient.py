import numpy as np
import matplotlib.pyplot as plt
from qiskit.primitives import Sampler
import qiskit
from qiskit import Aer, QuantumCircuit, transpile, assemble
import base.constant
import base.hamiltonian
def expected(circuit,params):
    """
      Expectation
      INPUT:
        circuit: quantum circuit
        shots : Number of hits
      OUTPUT:
        Returns measurement results
    
    """
    qc=circuit.copy()
    qc.measure_all()
    sampler = Sampler()
    result = sampler.run(qc,parameter_values=params, shots = 10000).result().quasi_dists[0].get(0, 0)
    
    return result
def loss_function(circuit,thetass):
    """
      Calculating list value loss function values
      INPUT:
        thetass: Parameter array
        circuit: quantum circuit
    
    """
    listloss=[]
    for i in thetass:
     expectation_value = expected(circuit,i)
     loss = 1-expectation_value
     listloss.append(loss)
    return loss

def parameter_shift_gradient(circuit,params, index):
    """
      Calculating derivatives using parameter-shift-rule
      INPUT:
        circuit: quantum circuit
        params: Parameter array
        index: the index under consideration of the array
      
        
    """
    params_plus = params.copy()
    params_minus = params.copy()


    params_plus[index] += base.constant.delta
    params_minus[index] -= base.constant.delta

    gradient = (expected(circuit,params_plus) - expected(circuit,params_minus)) / (2 * base.constant.delta)
    
    return gradient
def parameter_optimization(circuit,thetas):
    """
      Parameter optimization using parameter-shift-rule
      INPUT:
       circuit: quantum circuit
       Thetas: Parametric array
      OUTPUT:
       Returns array and parameter array list
    """
    initial_params = thetas
    learning_rate = base.constant.learning_rate
    max_iterations = base.constant.max_iterations
    thetass=[]
    for iteration in range(max_iterations):
        gradients = [parameter_shift_gradient(circuit,initial_params, i) for i in range(len(initial_params))]
        initial_params = initial_params - learning_rate * np.array(gradients)
        thetass.append(initial_params)
    return initial_params,thetass
