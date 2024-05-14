import numpy as np
from scipy.signal import find_peaks
from . import constant
from . import hamiltonian
from . import object


def Pmax(num_qubits, t1, thetas, shots):
    """
        Return Pmax

        INPUT:
          t: time
          thetas: array parameter (length 2*(N-1))
          shots: number of measurement times

    """
    time = np.linspace(0, t1, shots)
    arrayW = []
    arrayP = []
    for t in time:
        w = object.E(num_qubits, t, thetas) - object.E(num_qubits, 0, thetas)
        if (t != 0):
            p = w/t
        else:
            p = 0
        arrayW.append(w)
        arrayP.append(p)
    max_value = np.max(arrayP)
    return max_value


def grad_Pmax(num_qubits, tmax, thetas, h1, gamma = 0):
    psi_t = object.psi_t(num_qubits, tmax, h1)
    h0 = hamiltonian.h0(num_qubits)
    grad = np.zeros(len(thetas), dtype=np.complex128)
    for i in range(num_qubits - 1):
        Pi = (1 + gamma) * hamiltonian.Pi(num_qubits, 'XX', i) + (1 - gamma) * hamiltonian.Pi(num_qubits, 'YY', i)
        grad[i] = 1j * tmax * (np.transpose(np.conjugate(psi_t)) @ (np.transpose(np.conjugate(Pi)) @ h0 - h0 @ Pi) @ psi_t)[0,0]
    if len(thetas) > num_qubits - 1:
        for i in range(num_qubits - 1):
            Pi = hamiltonian.Pi(num_qubits, 'ZZ', i)  
            grad[i + num_qubits - 1] = 1j * tmax * (np.transpose(np.conjugate(psi_t)) @ (np.transpose(np.conjugate(Pi)) @ h0 - h0 @ Pi) @ psi_t)[0,0]
    return grad


def find_Pmax(num_qubits, h1, t = 2, delta_t = 0.01, auto_stop = True):
    Ps = []
    ts = np.arange(0, t, delta_t)
    for t in ts:
        P = object.P(num_qubits, t, h1)
        Ps.append(P)
        peaks, _ = find_peaks(Ps)
        if len(peaks) == 1 and auto_stop:
            break
    Pmax = np.max(Ps)
    tmax = ts[np.argmax(Ps)]
    return Ps, Pmax, tmax

