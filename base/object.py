import numpy as np
from . import hamiltonian
from . import constant
def E_1D(h1, t):
    """
        Return calculator E(t) = <psi(t)| H0 |psi(t)>
    """
    num_qubits = int(np.log2(h1.shape[0]))
    return np.real(np.transpose(np.conjugate(psi_t_1D(h1, t))) @ hamiltonian.h0_1D(num_qubits) @ psi_t_1D(h1, t))[0, 0]

def E_2D(h1, t):
    """
        Return calculator E(t) = <psi(t)| H0 |psi(t)>
    """
    # 
    num_qubits = int(np.log2(h1.shape[0]))
    k = psi_t_2D(h1, t)
    m = hamiltonian.h0_2D(num_qubits) 
    return np.real(np.transpose(np.conjugate(psi_t_2D(h1, t))) @ hamiltonian.h0_2D(num_qubits) @ psi_t_2D(h1, t))[0, 0]


def W_1D(h1, t):
    """
        Return W(t) = E(t) - E(0)
        INPUT:
          t: time
          thetas: array parameter (length 2*(N-1))
    """
    return E_1D(h1, t) - E_1D(h1, 0)


def P_1D(h1, t):
    """
        Return P(t) = W(t) / t
        INPUT:
          t: time
          thetas: array parameter (length 2*(N-1))
    """
    if (t == 0):
        return 0

    return W_1D(h1, t) / t

def W_2D(h1, t):
    """
        Return W(t) = E(t) - E(0)
        INPUT:
          t: time
          thetas: array parameter (length 2*(N-1))
    """
    k = E_2D(h1, t)
    m = E_2D(h1, 0)
    l = 1
    return E_2D(h1, t) - E_2D(h1, 0)


def P_2D(h1, t):
    """
        Return P(t) = W(t) / t
        INPUT:
          t: time
          thetas: array parameter (length 2*(N-1))
    """
    if (t == 0):
        return 0

    return W_2D(h1, t) / t



def psi_0_2D(num_qubits):
    """
       Return ps0 = dw \otimes dw \otimes ... dw   (have N dw)

    """

    psi_0 = constant.state
    for i in range(num_qubits - 1):
        psi_0 = np.kron(psi_0, constant.state)
    return psi_0

def psi_0_1D(num_qubits):
    """
       Return ps0 = dw \otimes dw \otimes ... dw   (have N dw)
    """
    psi_0 = constant.dw
    for i in range(num_qubits - 1):
        psi_0 = np.kron(psi_0, constant.dw)
    return psi_0


def psi_t_1D(h1, t):
    """
        Return psit = e^-i*t*H1 * psi_0(num_qubits)
        INPUT:

          t: time
          thetas: array parameter (length 2*(N-1))
    """
    num_qubits = int(np.log2(h1.shape[0]))
    psi_t = np.exp(-1j * t * h1) @ psi_0_1D(num_qubits)
    return psi_t

def psi_t_2D(h1, t):
    """
        Return psit = e^-i*t*H1 * psi_0(num_qubits)
        INPUT:

          t: time
          thetas: array parameter (length 2*(N-1))
    """
    num_qubits = int(np.log2(h1.shape[0]))
    psi_t = np.exp(-1j * t * h1) @ psi_0_2D(num_qubits)
    return psi_t

# def E(t, h1, h0, ps0):
#     """
#         Return calculator E(t) = <psi(t)| H0 |psi(t)>

#         INPUT:

#           thetas: array parameter (length 2*(N-1))
#     """
#     #
#     pt = np.exp(-1j * t * h1) @ ps0
#     return np.real((tq.daggx(pt)) @ h0 @ pt)[0, 0]

def Ps_1D(h1, t, delta_t):
    """

    """
    Ws = []
    Ps = []
    ts = np.arange(0, t, delta_t)

    for time in ts:

        if (time != 0):
            w = W_1D(h1, time)
            p = P_1D(h1, time)
        else:
            w = 0
            p = 0
        Ws.append(w)
        Ps.append(p)
    Pmax = np.max(Ps)
    tmax = ts[np.argmax(Ps)]
    return Ps, Pmax, tmax



def Ps_2D(h1, t, delta_t):
    """

    """
    Ws = []
    Ps = []
    ts = np.arange(0, t, delta_t)
    for time in ts:

        if (time != 0):
            w = W_2D(h1, time)
            p = P_2D(h1, time)
        else:
            w = 0
            p = 0
        Ws.append(w)
        Ps.append(p)
    Pmax = np.max(Ps)
    tmax = ts[np.argmax(Ps)]
    return Ps, Pmax, tmax
