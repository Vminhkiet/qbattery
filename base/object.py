import numpy as np
import base.hamiltonian
import base.constant
from scipy.linalg import expm
import tqix as tq

def psi_0(num_qubits):
    """
       Return ps0 = dw \otimes dw \otimes ... dw   (have N dw)

    """
    psi_0 = base.constant.dw
    for i in range(num_qubits - 1):
        psi_0 = np.kron(psi_0 , base.constant.dw)
    return psi_0

def psi_t(num_qubits, t, thetas):
    """
        Return psit = e^-i*t*H1 * psi_0(num_qubits)
        INPUT:

          t: time
          thetas: array parameter (length 2*(N-1))
    """
    psi_t = expm(-1j * t * base.hamiltonian.h1(num_qubits, thetas))@ psi_0(num_qubits)
    return psi_t 

def E(num_qubits, t, thetas):
    """
        Return calculator E(t) = <psi(t)| H0 |psi(t)>

        INPUT:

          thetas: array parameter (length 2*(N-1))
    """
    return np.real(( tq.daggx(psi_t(num_qubits,t,thetas)) @ base.hamiltonian.h0(num_qubits) @ psi_t(num_qubits,t,thetas) )[0,0])



        
