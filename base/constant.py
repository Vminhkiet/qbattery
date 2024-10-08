import numpy as np

learning_rate = 0.3
delta = 0.001
max_iterations = 10

h = 1

Y = np.array([[0, -1j],[1j, 0]])
X = np.array([[0, 1],[1, 0]])
Z = np.array([[1, 0],[0, -1]])
I = np.eye(2)
H = 1/np.sqrt(2)*np.array([[1, 1],[1, -1]])
XX = np.kron(X,X)
YY = np.kron(Y,Y)
ZZ = np.kron(Z,Z)
dw = np.array([[1/np.sqrt(2)],[-1j/np.sqrt(2)]])
state=np.array([[1/np.sqrt(2)],[1/np.sqrt(2)]])
tubular = 1
row = 1