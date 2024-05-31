import numpy as np
from scipy.signal import find_peaks
from . import constant
from . import hamiltonian
from . import object





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
            Pi = hamiltonian.Pi_thetasG_2D(n_row, n_col, 'Z', i)
        else:
            Pi = hamiltonian.Pi_thetasJ_2D(n_row, n_col, 'Z', i)

        grad[i] = 1j * tmax * (np.transpose(np.conjugate(psi_t)) @
                               (np.transpose(np.conjugate(Pi)) @ h0 - h0 @ Pi) @ psi_t)[0, 0]

    return grad
