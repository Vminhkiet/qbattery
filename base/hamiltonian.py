import numpy as np
from qiskit.quantum_info import Operator
from scipy.linalg import expm
from . import constant

def Pi_1D(num_qubits, index):
    sum = constant.YY
    if (index % 2 == 0):
        sum = constant.XX

    for _ in range(num_qubits-index-2):
        sum = np.kron(constant.I, sum)

    for _ in range(num_qubits-index + 1, num_qubits):
        sum = np.kron(sum, constant.I)
    return sum


def h0_1D(num_qubits):
    """
        Return H0 is formula in photo of teacher BINHO
    """
    sum_h0 = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    for i in range(num_qubits):
        H0 = constant.Y
        for j in range(i):
            H0 = np.kron(constant.I, H0)
        for j in range(i+1, num_qubits):
            H0 = np.kron(H0, constant.I)
        sum_h0 += H0
    return sum_h0 * constant.h


def h0_2D(num_qubits, h = 1):
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


def Pij_2D(n_rows, n_columns, term, i, j):
    num_qubits = n_columns*n_rows
    # điều kiện nếu ô không hợp lệ trả về 0
    # i,j bây giờ đang ở dạng ma trận (row+2)x(column+2) đếm theo thứ tự từ trái sang phải từ trên xuống dưới
    if (i < n_columns+2 or j < n_columns+2 or i % (n_columns+2) == 0 or j % (n_columns+2) == 0 or (i+1) % (n_columns+2) == 0 or (j+1) % (n_columns+2) == 0 or i >= (n_columns+2)*(n_rows+1) or j >= (n_columns+2)*(n_rows+1)):
        return 0

    # chuyển về i,j đếm ở dạng ma trận row x column đếm theo cách tương tự để tính ZiZj
    i = (i//(n_columns+2)-1)*n_columns+i % (n_columns+2)-1
    j = (j//(n_columns+2)-1)*n_columns+j % (n_columns+2)-1

    x = i
    y = j

    if (j < i):
        x = j
        y = i

    # vd công thức I@Z1@I@I@Z4@I thì tạo P=I@I => P=Z1@P@Z4 => I@P@I
    P = constant.I
    #PS = 'I'
    if (x+1 == y):
        P = 1
        #PS = ''
    
    for _ in range(x+1, y-1):
        P = np.kron(P, constant.I)
        #PS += 'I'
    if term == 'Z':
        P = np.kron(constant.Z, np.kron(P, constant.Z))
        #PS += 'Z' + PS + 'Z'
    elif term == 'X':
        P = np.kron(constant.X, np.kron(P, constant.X))
        #PS += 'X' + PS + 'X'
    elif term == 'Y':
        P = np.kron(constant.Y, np.kron(P, constant.Y))
        #PS += 'Y' + PS + 'Y'
    for _ in range(x):
        P = np.kron(constant.I, P)
        # PS = 'I' + PS
    for _ in range(y+1, num_qubits):
        P = np.kron(P, constant.I)
    #     PS = PS + 'I'
    # print('Symbol: ', PS)
    # print('Dim Pij: ', len(P))
    return P


def Pi_thetasJ_2D(n_row, n_col, term, i):
    num_qubits = n_row * n_col
    Pi = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    # chuyển về dạng hàng x, cột y của ma trận row x col
    x = i // n_col + 1
    y = i % n_col + 1
    # tiếp đến xét ma trận (row+2)x(col+2) nếu là ô không thõa mãn thì trả về 0
    Pi = Pi + Pij_2D(n_row, n_col, term, x*(n_col+2)+y, x*(n_col + 2)+y+1)
    Pi = Pi + Pij_2D(n_row, n_col, term, x*(n_col+2)+y, x*(n_col + 2)+y-1)
    Pi = Pi + Pij_2D(n_row, n_col, term, x*(n_col+2)+y, (x+1)*(n_col + 2)+y)
    Pi = Pi + Pij_2D(n_row, n_col, term, x*(n_col+2)+y, (x-1)*(n_col + 2)+y)
    return Pi


def Pi_thetasG_2D(n_row, n_col, term, i):
    num_qubits = n_row * n_col
    Pi = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    # chuyển về dạng hàng x, cột y của ma trận row x col
    x = i // n_col + 1
    y = i % n_col + 1
    # tiếp đến xét ma trận (row+2)x(col+2) nếu là ô không thõa mãn thì trả về 0
    Pi = Pi + Pij_2D(n_row, n_col, term, x*(n_col+2)+y, (x-1)*(n_col+2)+y-1)
    Pi = Pi + Pij_2D(n_row, n_col, term, x*(n_col+2)+y, (x+1)*(n_col+2)+y+1)
    Pi = Pi + Pij_2D(n_row, n_col, term, x*(n_col+2)+y, (x+1)*(n_col+2)+y-1)
    Pi = Pi + Pij_2D(n_row, n_col, term, x*(n_col+2)+y, (x-1)*(n_col+2)+y+1)
    return Pi


def h1_2D(n_rows, n_columns, thetas, h = 0):
    num_qubits = n_columns*n_rows
    # H0 = h0_2D(num_qubits, h)
    H1 = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    # ô hợp lệ là ô nằm trong ma trận row x column khi thêm trái phải trên dưới 1 hàng cột cho ma trận row x column thì những ô được thêm là ô không hợp lệ
    # đang xét những ô hợp lệ theo ma trận row x col
    for i in range(num_qubits):
        H1 = H1 + Pi_thetasJ_2D(n_rows, n_columns, 'Z', i)*thetas[i]
        H1 = H1 + Pi_thetasG_2D(n_rows, n_columns, 'Z', i)*thetas[i+num_qubits]
    return H1 # H1 + H0


def h_general_1D(num_qubits, thetas, gamma, h):
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
            hamiltonian += - \
                thetas[i]*((1 + gamma)*XX + (1 - gamma)*YY) - \
                thetas[i+num_qubits-1] * ZZ - h * Z
        elif thetas.shape[0] == num_qubits-1:
            hamiltonian += -thetas[i]*((1 + gamma)*XX + (1 - gamma)*YY) - h * Z
    return hamiltonian


def h1_xx_1D(num_qubits, thetas):
    if len(thetas) != num_qubits-1:
        raise ValueError('The number of parameters is not correct')
    return h_general_1D(num_qubits, thetas, gamma=0, h=0)


def h1_xy_1D(num_qubits, thetas, gamma):
    if len(thetas) != 2*(num_qubits-1):
        raise ValueError('The number of parameters is not correct')
    return h_general_1D(num_qubits, thetas, gamma, h=0)


def h1_xxz_1D(num_qubits, thetas):
    if len(thetas) != 2*(num_qubits-1):
        raise ValueError('The number of parameters is not correct')
    return h_general_1D(num_qubits, thetas, gamma=0, h=0)


def h1_1D(num_qubits, thetas):
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
