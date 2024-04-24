import numpy as np
import base.hamiltonian
import base.constant
from scipy.linalg import expm

def psi_0(num_qubits):
    """
       Return ps0 = dw \otimes dw \otimes ... dw   (have N dw)

    """
    psi_0 = base.constant.dw
    for i in range(num_qubits - 1):
        psi_0 = np.kron(psi_0, base.constant.dw)
    return psi_0


def psi_t(num_qubits, t, h1):
    """
        Return psit = e^-i*t*H1 * psi_0(num_qubits)
        INPUT:

          t: time
          thetas: array parameter (length 2*(N-1))
    """
    psi_t = expm(-1j * t * h1) @ psi_0(num_qubits)
    return psi_t


def E(num_qubits, t, h1):
    """
        Return calculator E(t) = <psi(t)| H0 |psi(t)>

        INPUT:

          thetas: array parameter (length 2*(N-1))
    """
    # 
    return np.real(np.transpose(np.conjugate(psi_t(num_qubits, t, h1))) @ base.hamiltonian.h0(num_qubits) @ psi_t(num_qubits, t, h1))[0, 0]


def W(num_qubits, t, h1):
    """
        Return W(t) = E(t) - E(0)

        INPUT:

          t: time
          thetas: array parameter (length 2*(N-1))
    """
    return E(num_qubits, t, h1) - E(num_qubits, 0, h1)


def P(num_qubits, t, h1):
    """
        Return P(t) = W(t) / t

        INPUT:

          t: time
          thetas: array parameter (length 2*(N-1))
    """
    if (t == 0):
        return 0

    return W(num_qubits, t, h1) / t
