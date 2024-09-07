import base.object
import base.hamiltonian
import base.Gradient
import numpy as np
import tqix as tq
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

point_graph = 3
z = 2
n_col,n_row = base.hamiltonian.covert_3D_to_2D(point_graph,z)
num_qubits = n_row * n_col
thetas = np.random.rand(2 * num_qubits)
thetas = thetas.astype(np.complex128)
h1 = base.hamiltonian.h1_2D(n_row, n_col, thetas)
print(h1)
t = 6
delta_t = 0.01
import numpy as np
from scipy.linalg import expm

# Giá trị của t
t = 1.0

# Tính toán ma trận mũ
matrix_exponential_simple = expm(-1j * t * h1)

print("Ma trận mũ của ma trận h1 là:")
print(matrix_exponential_simple.shape)
import base.circuit

thetas1,thetass=base.Gradient.parameter_optimization(n_row,n_col,t,thetas)
print(thetas1)