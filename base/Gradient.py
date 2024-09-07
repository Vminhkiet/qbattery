import numpy as np
from scipy.signal import find_peaks
from . import constant
from . import hamiltonian
from . import object
from . import circuit
from qiskit.primitives import Sampler




def grad_Pmax_1D(tmax, thetas, h1, gamma=0):
    num_qubits = int(np.log2(h1.shape[0]))
    psi_t = object.psi_t_1D(tmax, h1)
    h0 = hamiltonian.h0_1D(num_qubits)
    grad = np.zeros(len(thetas), dtype=np.complex128)
    for i in range(num_qubits - 1):
        Pi = (1 + gamma) * hamiltonian.Pi(num_qubits, 'XX', i) + \
            (1 - gamma) * hamiltonian.Pi(num_qubits, 'YY', i)
        grad[i] = 1j * tmax * (np.transpose(np.conjugate(psi_t)) @
                               (np.transpose(np.conjugate(Pi)) @ h0 - h0 @ Pi) @ psi_t)[0, 0]
    if len(thetas) > num_qubits - 1:
        for i in range(num_qubits - 1):
            Pi = hamiltonian.Pi(num_qubits, 'ZZ', i)
            grad[i + num_qubits - 1] = 1j * tmax * (np.transpose(np.conjugate(psi_t)) @ (
                np.transpose(np.conjugate(Pi)) @ h0 - h0 @ Pi) @ psi_t)[0, 0]
    return grad


def find_Pmax_1D(h1, t=2, delta_t=0.01, auto_stop=True):
    Ps = []
    ts = np.arange(0, t, delta_t)
    for t in ts:
        P = object.P_1D(h1, t)
        print(P)
        Ps.append(P)
        peaks, _ = find_peaks(Ps)
        if len(peaks) == 1 and auto_stop:
            break
    Pmax = np.max(Ps)
    tmax = ts[np.argmax(Ps)]
    return Ps, Pmax, tmax


def find_Pmax_2D(h1, t=2, delta_t=0.01, auto_stop=True):
    Ps = []
    ts = np.arange(0, t, delta_t)
    for t in ts:
        P = object.P_2D(h1, t)
        Ps.append(P)
        peaks, _ = find_peaks(Ps)
        if len(peaks) == 1 and auto_stop:
            break
    Pmax = np.max(Ps)
    tmax = ts[np.argmax(Ps)]
    return Ps, Pmax, tmax


def grad_Pmax_2D(n_row, n_col, tmax, thetas, h1):
    num_qubits = n_row * n_col
    psi_t = object.psi_t_2D(h1, tmax)
    h0 = hamiltonian.h0_2D(num_qubits)
    grad = np.zeros(len(thetas), dtype=np.complex128)
    Pi = 0
    for i in range(2*num_qubits):

        if (i >= num_qubits):
            Pi = hamiltonian.Pi_thetasG_2D(n_row, n_col, 'X', i)
        else:
            Pi = hamiltonian.Pi_thetasJ_2D(n_row, n_col, 'X', i)

        grad[i] = 1j * tmax * (np.transpose(np.conjugate(psi_t)) @
                               (np.transpose(np.conjugate(Pi)) @ h0 - h0 @ Pi) @ psi_t)[0, 0]

    return grad
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
    N=qc.num_qubits
    qc.measure_all()
    sampler = Sampler()
    result = sampler.run(qc,parameter_values=params, shots = 100).result().quasi_dists[0].get(2**N-1,0)
    return result
def find_tmax(n_row, n_col,t,thetas):
    delta_t = 0.01
    h1 = hamiltonian.h1_2D(n_row, n_col, thetas)
    Ps, Pmax, tmax = object.Ps_2D(h1, t, delta_t)
    return tmax
def loss_function(n_row, n_col,t,thetas,alpha=1,beta=1):
    """
      Calculating list value loss function values
      INPUT:
        thetass: Parameter array
        circuit: quantum circuit
    
    """
    qc,ts=circuit.aqc(n_row*n_col, t, hamiltonian.h1_2D(n_row, n_col, thetas))
    expectation_value = expected(qc,ts)
    #tmax = find_tmax(n_row, n_col,t,thetas)
    loss = beta*(1-expectation_value) + alpha*t
    return loss
def parameter_shift_gradient(n_row, n_col,t,params, index):
    """
      Calculating derivatives using parameter-shift-rule
      INPUT:
        circuit: quantum circuit
        params: Parameter array
        index: the index under consideration of the array
      
        
    """
    params_plus = params.copy()
    params_minus = params.copy()


    params_plus[index] += constant.delta
    params_minus[index] -= constant.delta
    
    gradient = (loss_function(n_row, n_col,t,params_plus) - loss_function(n_row, n_col,t,params_minus)) / (2 * constant.delta)

    return gradient
def parameter_optimization(n_row, n_col,t,thetas):
    """
      Parameter optimization using parameter-shift-rule
      INPUT:
       circuit: quantum circuit
       Thetas: Parametric array
      OUTPUT:
       Returns array and parameter array list
    """
    initial_params = thetas
    learning_rate = constant.learning_rate
    max_iterations = constant.max_iterations
    thetass=[]
    for iteration in range(max_iterations):
        gradients = [parameter_shift_gradient(n_row, n_col,t,initial_params, i) for i in range(len(initial_params))]
        initial_params = initial_params - learning_rate * np.array(gradients)
        thetass.append(initial_params)
    return initial_params,thetass