import numpy as np
import base.hamiltonian
import base.constant

import tqix as tq


def psi_0(num_qubits):
    """
       Return ps0 = dw \otimes dw \otimes ... dw   (have N dw)

    """

    psi_0 = base.constant.state
    for i in range(num_qubits - 1):
        psi_0 = np.kron(psi_0, base.constant.state)
    return psi_0


def psi_t(num_qubits, t, h1):
    """
        Return psit = e^-i*t*H1 * psi_0(num_qubits)
        INPUT:

          t: time
          thetas: array parameter (length 2*(N-1))
    """

    psi_t = np.exp(-1j * t * h1) @ psi_0(num_qubits)
    return psi_t


def E(t, h1, h0, ps0):
    """
        Return calculator E(t) = <psi(t)| H0 |psi(t)>

        INPUT:

          thetas: array parameter (length 2*(N-1))
    """
    #
    pt = np.exp(-1j * t * h1) @ ps0
    return np.real((tq.daggx(pt)) @ h0 @ pt)[0, 0]
