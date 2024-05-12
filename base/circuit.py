import matplotlib.pyplot as plt
import tqix as tq
from tqix import *
import numpy as np, qiskit
from qiskit.quantum_info import Operator
from scipy.linalg import expm
from . import hamiltonian
from qsee.compilation.qsp import QuantumStatePreparation
from qiskit.quantum_info import Statevector
def ansatz(num_qubits,t,thetas):
    """
        Input:

          t:time
          H1:hamiltonian operator

        Output:

          Return quantum circuit representing the operation e^-(i*t*H1)
    """
    u = expm(-1j*t*hamiltonian.h1(num_qubits,thetas))
    qst=QuantumStatePreparation.prepare(u)
    ansatz = qst.u
    ttas=qst.thetas
    return ansatz,ttas


def circuit_Rx(num_qubits):
    """
      Input:
      Output:
      
        Returns the quantum circuit with all qubits given each qubit an additional Rx gate
    """
    qc = QuantumCircuit(num_qubits)
    qc.rx(np.pi/2,range(num_qubits))
    return qc


def aqc(num_qubits,t,thetas):
    """
        Input:

          t:time
          H1:hamiltonian operator

        Output:

          Return quantum circuit = circuit_Rx + ansatz
    """

    qc1 = circuit_Rx(num_qubits)
    ansatz1,ttas = ansatz(num_qubits,t,thetas)
    qc1.compose(ansatz1,range(num_qubits),inplace=True)

    return qc1,ttas