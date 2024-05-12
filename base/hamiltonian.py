import numpy as np
from qiskit.quantum_info import Operator
from scipy.linalg import expm
from . import constant

# def Pi(num_qubits, index):
#     sum = constant.YY
#     if (index % 2 == 0):
#         sum = constant.XX

#     for _ in range(num_qubits-index-2):
#         sum = np.kron(constant.I, sum)

#     for _ in range(num_qubits-index + 1, num_qubits):
#         sum = np.kron(sum, constant.I)
#     return sum


# def h0(num_qubits, h):
#     qc = circuit(num_qubits)
#     Jz = qc.Jz()
#     return (h * Jz).todense()

def h0(num_qubits,h):
    """
        Return H0 is formula in photo of teacher BINHO

    """
    sum_h0 = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    for i in range(num_qubits):
        H0 = constant.X
        for j in range(i):
            H0 = np.kron(constant.I, H0)
        for j in range(i+1, num_qubits):
            H0 = np.kron(H0, constant.I)
        sum_h0 += H0
    return sum_h0*-h


def Pi(num_qubits, term, index):
    """_summary_

    Args:
        num_qubits (int): _description_
        term (str): XX,YY,ZZ,Z
        index (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Single term
    
    # Double term
    if term == 'XX':
        P = constant.XX
    elif term == 'YY':
        P = constant.YY
    elif term == 'ZZ':
        P = constant.ZZ
    elif term == 'Z':
        P = constant.Z
    elif term == 'X':
        P = constant.X
    elif term == 'Y':
        P = constant.Y
    for _ in range(num_qubits-index-len(term)):
        P = np.kron(constant.I, P)
    for _ in range(index):
        P = np.kron(P, constant.I)
    return P
def Pij(n_rows,n_columns, term, i,j):
    num_qubits=n_columns*n_rows

    if(i<n_columns+2 or j<n_columns+2 or i%(n_columns+2)==0 or j%(n_columns+2)==0 or (i+1) % (n_columns+2) == 0 or (j+1) % (n_columns+2)==0 or i>=(n_columns+2)*(n_rows+1) or j>=(n_columns+2)*(n_rows+1)):
        return 0
    
    
    i=(i//(n_columns+2)-1)*n_columns+i%(n_columns+2)-1
    j=(j//(n_columns+2)-1)*n_columns+j%(n_columns+2)-1
    print(i,j)
    x=i
    y=j

    if(j<i):
        x=j
        y=i

    P= constant.I
    if(x+1==y):
        P = 1
    for _ in range(x+1,y-1):
        P = np.kron(P,constant.I)
    if term == 'Z':
        P = np.kron(constant.Z,np.kron(P,constant.Z))
    elif term == 'X':
        P = np.kron(constant.X,np.kron(P,constant.X))
    elif term == 'Y':
        P = np.kron(constant.Y,np.kron(P,constant.Y))
    for _ in range(x):
        P = np.kron(constant.I, P)
    for _ in range(y+1,num_qubits):
        P = np.kron(P, constant.I)
    return P


def h1_2D(n_rows,n_columns,thetasJ,thetasG,h):
    num_qubits=n_columns*n_rows
    H0=h0(num_qubits,h)
    H1=np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)

    for i in range(1,n_rows+1):
        for j in range(1,n_columns+1):

            H1 = H1 + Pij(n_rows,n_columns,'Z',i*(n_columns+2)+j,i*(n_columns+2)+j+1) * thetasJ[(i-1)*n_columns+j-1]
            H1 = H1 + Pij(n_rows,n_columns,'Z',i*(n_columns+2)+j,i*(n_columns+2)+j-1) * thetasJ[(i-1)*n_columns+j-1]
            H1 = H1 + Pij(n_rows,n_columns,'Z',i*(n_columns+2)+j,(i+1)*(n_columns+2)+j) * thetasJ[(i-1)*n_columns+j-1]
            H1 = H1 + Pij(n_rows,n_columns,'Z',i*(n_columns+2)+j,(i-1)*(n_columns+2)+j) * thetasJ[(i-1)*n_columns+j-1]
            H1 = H1 + Pij(n_rows,n_columns,'Z',i*(n_columns+2)+j,(i-1)*(n_columns+2)+j-1) * thetasG[(i-1)*n_columns+j-1]
            H1 = H1 + Pij(n_rows,n_columns,'Z',i*(n_columns+2)+j,(i+1)*(n_columns+2)+j+1) * thetasG[(i-1)*n_columns+j-1]
            H1 = H1 + Pij(n_rows,n_columns,'Z',i*(n_columns+2)+j,(i+1)*(n_columns+2)+j-1) * thetasG[(i-1)*n_columns+j-1]
            H1 = H1 + Pij(n_rows,n_columns,'Z',i*(n_columns+2)+j,(i-1)*(n_columns+2)+j+1) * thetasG[(i-1)*n_columns+j-1]

            
    return H0+H1

def h_general(num_qubits, thetas, gamma, h):
    """
        Return H1 is formula in file quantum battery 2

        INPUT:

          thetas: parameter with number 2*(N-1)
    """
    hamiltonian = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    # XX spin down
    for i in range(num_qubits-1):
        XX = Pi(num_qubits, 'XX', i)
        YY = Pi(num_qubits, 'YY', i)
        ZZ = Pi(num_qubits, 'ZZ', i)
        Z = Pi(num_qubits, 'Z', i)
        if thetas.shape[0] == 2*(num_qubits-1):
            hamiltonian += -thetas[i]*((1 + gamma)*XX + (1 - gamma)*YY) - thetas[i+num_qubits-1] * ZZ - h * Z
        elif thetas.shape[0] == num_qubits-1:
            hamiltonian += -thetas[i]*((1 + gamma)*XX + (1 - gamma)*YY) - h * Z
    return hamiltonian


def h1_xx(num_qubits, thetas):
    if len(thetas) != num_qubits-1:
        raise ValueError('The number of parameters is not correct')
    return h_general(num_qubits, thetas, gamma = 0, h = 0)

def h1_xy(num_qubits, thetas, gamma):
    if len(thetas) != 2*(num_qubits-1):
        raise ValueError('The number of parameters is not correct')
    return h_general(num_qubits, thetas, gamma, h = 0)

def h1_xxz(num_qubits, thetas):
    if len(thetas) != 2*(num_qubits-1):
        raise ValueError('The number of parameters is not correct')
    return h_general(num_qubits, thetas, gamma=0 , h = 0)

            



def h1(num_qubits, thetas):
    """
        Return H1 is formula in file quantum battery 2

        INPUT:

          thetas: parameter with number 2*(N-1)
    """
    hamiltonian = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    for i in range(num_qubits-1):
        sumx = constant.XX
        sumy = constant.YY

        for _ in range(num_qubits-i-2):
            sumx = np.kron(constant.I, sumx)
            sumy = np.kron(constant.I, sumy)

        for _ in range(num_qubits-i, num_qubits):
            sumx = np.kron(sumx, constant.I)
            sumy = np.kron(sumy, constant.I)

        hamiltonian += thetas[2*i] * sumx + thetas[(2*i)+1] * sumy

    return hamiltonian
