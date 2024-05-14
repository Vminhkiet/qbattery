import numpy as np
from scipy.signal import find_peaks
from . import constant
from . import hamiltonian
from . import object
import tqix as tq


def Pmax(num_qubits, t1, h1, shots):
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
    h0=hamiltonian.h0(num_qubits,h=1)
    ps0=object.psi_0(num_qubits)
    E0=np.real( (tq.daggx(ps0)) @h0@ ps0)[0, 0]
    print(E0)
    for t in time:
        w = 0
        p = 0
        if(t!=0):
            w = object.E(t,h1,h0,ps0) - E0
            p = w/t
        
        print(w)
        arrayW.append(w)
        arrayP.append(p)
    max_value = np.max(arrayP)
    tmax=time[np.argmax(arrayP)]
    return arrayP,max_value,tmax


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

def grad_2D_Pmax(n_row,n_col, tmax, thetas, h1):
    num_qubits = n_row * n_col
    psi_t = object.psi_t(num_qubits, tmax, h1)
    h0 = hamiltonian.h0(num_qubits,h=1)
    grad = np.zeros(len(thetas), dtype=np.complex128)
    Pi = 0
    for i in range(2*num_qubits):
        
        if(i>=num_qubits):
            Pi=hamiltonian.Pi_thetasg_2d(n_row,n_col,'Z',i)
        else:
            Pi=hamiltonian.Pi_thetasj_2d(n_row,n_col,'Z',i)

        grad[i] = 1j * tmax * (np.transpose(np.conjugate(psi_t)) @ (np.transpose(np.conjugate(Pi)) @ h0 - h0 @ Pi) @ psi_t)[0,0]
   

    return grad