import numpy as np

learning_rate = 0.3
delta = 0.001
<<<<<<< HEAD
max_iterations = 150
=======
max_iterations = 100
>>>>>>> b3f939dd7cf083d5e26d18ffeceb42b55c60af3d

h = 1

Y = np.array([[0, -1j],[1j, 0]])
X = np.array([[0, 1],[1, 0]])
I = np.eye(2)

XX = np.kron(X,X)
YY = np.kron(Y,Y)
dw = np.array([[1/np.sqrt(2)],[-1j/np.sqrt(2)]])