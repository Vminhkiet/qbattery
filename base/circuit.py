import matplotlib.pyplot as plt
import tqix as tq
from tqix import *
import numpy as np, qiskit
from qiskit.quantum_info import Operator
from scipy.linalg import expm
from . import hamiltonian
from qoop.compilation.qsp import QuantumStatePreparation
from qiskit.quantum_info import Statevector
def ansatz(t,h1):
    """
        Input:

          t:time
          H1:hamiltonian operator

        Output:

          Return quantum circuit representing the operation e^-(i*t*H1)
    """
    u = expm(-1j*t*h1)
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


def aqc(num_qubits,t,h1):
    """
        Input:

          t:time
          H1:hamiltonian operator

        Output:

          Return quantum circuit = circuit_Rx + ansatz
    """

    qc1 = circuit_Rx(num_qubits)
    ansatz1,ts = ansatz(t,h1)
    qc1.compose(ansatz1,range(num_qubits),inplace=True)
    print("aasd")
    return qc1,ts