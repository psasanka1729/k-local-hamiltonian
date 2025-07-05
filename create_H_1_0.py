import sys
import math
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.linalg import eigh
from scipy.sparse import identity, csr_matrix, kron, save_npz, load_npz
from k_local_hamiltonian import KLocalHamiltonian


L = 10
k_lst = [2, 3, 4, 5, 6, 7]

index = int(sys.argv[1])
k = k_lst[index]

adjacent_number = ( L - k + 1 ) * 3 ** k
nonadjacent_number = math.comb(L, k) * 3 ** k

print('L =', L, 'k =', k)
print('Number of adjacent Pauli strings =', adjacent_number)
print('Number of non adjacent Pauli strings = ', nonadjacent_number)

mu = 1.0
sigma_strength = 0.0

random_coefficients = np.random.normal(mu, sigma_strength, nonadjacent_number)
save_npz(f'H_1_0_random_coefficients_L_{L}_k_{k}.npz', csr_matrix(random_coefficients))

# Initialize the Hamiltonian generator
hamiltonian_generator = KLocalHamiltonian(L = L, k = k, random_coefficients = random_coefficients, adjacent = False)
H_1_0 = hamiltonian_generator.hamiltonian_matrix()
save_npz(f'H_1_0_L_{L}_k_{k}_sigma_{sigma_strength}.npz', H_1_0)

