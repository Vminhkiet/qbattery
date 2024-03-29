import matplotlib.pyplot as plt
import tqix as tq
from tqix import *
import numpy as np, qiskit
from qiskit.quantum_info import Operator
import base.Gradient as gr
from scipy.linalg import expm
import base.constant as ct
def h0():
    sum_h0 = np.zeros((2**ct.N, 2**ct.N), dtype=complex)
    for i in range(ct.N):
        H0=ct.Y
        for j in range(i):
           H0=np.kron(ct.I,H0)
        for j in range(i+1,ct.N):
           H0=np.kron(H0,ct.I)
        sum_h0 += H0
    return sum_h0 * ct.h
def h1(thetas):
    # What is H1=GA
    hamiltonian = np.zeros((2**ct.N, 2**ct.N), dtype=complex)
    for i in range(ct.N-1):
        sumx = ct.XX
        sumy = ct.YY
        for _ in range(ct.N-i-2):
            sumx = np.kron(ct.I, sumx)
            sumy = np.kron(ct.I, sumy)
        for _ in range(ct.N-i, ct.N):
            sumx = np.kron(sumx, ct.I)
            sumy = np.kron(sumy, ct.I)
        hamiltonian += thetas[2*i] * sumx + thetas[(2*i)+1] * sumy
    
    return hamiltonian