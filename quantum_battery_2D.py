import matplotlib.pyplot as plt
import base.gradient
import base.object
import base.hamiltonian
import numpy as np
import tqix as tq


t = 2
delta_t = 0.001
n_row = 2
n_column = 2
num_qubits = n_row * n_column

# ltime, arP, arW = test(num_qubits, t, h1, shots=1024)

# max_value = np.max(arP)
# max_index = np.argmax(arP)
# max_time = ltime[max_index]
thetas = np.random.uniform(0, 2*np.pi, 2 * num_qubits)
h1 = base.hamiltonian.h1_2D(n_row, n_column, thetas)
Ps, Pmax, tmax = base.object.Ps_2D(h1, t, delta_t)
plt.plot(np.arange(0,t, delta_t), Ps, label="Power")
plt.xlabel("Time")
plt.ylabel("Value")
#plt.scatter(max_time, max_value, color="red", label=f"Max: {np.round(max_value, 2)}")
plt.legend()
plt.show()