import matplotlib.pyplot as plt
import tqix as tq
from tqix import *
import numpy as np, qiskit
from qiskit.quantum_info import Operator
from scipy.linalg import expm
import base.constant as ct

def pi(index,num_qubits):
    sum = ct.YY
    if(index%2==0):
        sum=ct.XX

    for _ in range(num_qubits-index-2):
        sum = np.kron(ct.I, sum)
        
    for _ in range(num_qubits-index + 1, num_qubits):
        sum = np.kron(sum, ct.I)
    return sum

def h0(num_qubits):
    """
        Return H0 is formula in photo of teacher BINHO

    """
    sum_h0 = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    for i in range(num_qubits):
        H0 = ct.Y
        for j in range(i):
           H0 = np.kron(ct.I,H0)
        for j in range(i+1,num_qubits):
           H0 = np.kron(H0,ct.I)
        sum_h0 += H0
    return sum_h0 * ct.h


def h1(num_qubits, thetas):
    """
        Return H1 is formula in file quantum battery 2

        INPUT:

          thetas: parameter with number 2*(N-1)
    """ 
    hamiltonian = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    for i in range(num_qubits-1):
        sumx = ct.XX
        sumy = ct.YY

        for _ in range(num_qubits-i-2):
            sumx = np.kron(ct.I, sumx)
            sumy = np.kron(ct.I, sumy)
            
        for _ in range(num_qubits-i, num_qubits):
            sumx = np.kron(sumx, ct.I)
            sumy = np.kron(sumy, ct.I)

        hamiltonian += thetas[2*i] * sumx + thetas[(2*i)+1] * sumy
    
    return hamiltonian