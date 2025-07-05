import sys
import numpy as np
import warnings
from scipy.linalg import eigh
from scipy.sparse import identity, csc_matrix, csr_matrix, kron, save_npz, load_npz

L_lst = [7, 8, 9, 10, 11, 12, 13]
k_lst = [2, 3]

save_npz('L_lst.npz', csr_matrix(L_lst))
save_npz('k_lst.npz', csr_matrix(k_lst))

mu = 1.0
sigma = 1.0
# Save the mu and sigma values to a file
save_npz('mu_sigma.npz', csr_matrix([mu, sigma]))

# Entanglement entropy calculation
def one_roll_operator(number_of_qubits: int):
     """
     Constructs the sparse matrix representation of the one-qubit roll operator.
     This operator cyclically permutes the qubits in the computational basis.

     Args:
          number_of_qubits (int): The total number of qubits (L) in the system.

     Returns:
          scipy.sparse.csc_matrix: The sparse rolling operator (R).
     """
     
     hilbert_space_dim = 2**number_of_qubits

     def bin_to_dec(binary_number_str: str) -> int:
          """Converts a binary string to its decimal integer representation."""
          return int(binary_number_str, 2)

     def dec_to_bin(decimal_number: int) -> str:
          """
          Converts a decimal integer to its binary string representation,
          padded with leading zeros to match the number of qubits.
          """
          # Convert to binary and remove the '0b' prefix
          binary_str = bin(decimal_number).replace("0b", "")
          
          # Add leading zeros to ensure the string has length `number_of_qubits`
          while len(binary_str) < number_of_qubits:
               binary_str = '0' + binary_str
          return binary_str

     def roll_string_once(binary_str: str) -> str:
          """
          Performs a single cyclic shift on the binary string.
          The last character becomes the first.
          """
          return binary_str[-1] + binary_str[:-1]

     # Use COO format for efficient construction of the sparse matrix
     rows = []
     cols = []
     data = []
     
     # For each basis state |i>, find where the roll operator sends it.
     # R|i> = |j> means R_ji = 1.
     for i in range(hilbert_space_dim):
          # 1. Convert the basis state index to its binary representation.
          binary_representation = dec_to_bin(i)
          
          # 2. Roll the qubits once.
          rolled_binary = roll_string_once(binary_representation)
          
          # 3. Convert the new binary representation back to a decimal index.
          j = bin_to_dec(rolled_binary)
          
          # The operator maps state |i> to state |j>.
          # In the matrix R, this corresponds to setting the element R[j, i] to 1.
          rows.append(j)
          cols.append(i)
          data.append(1)
          
     # Create the sparse matrix in Compressed Sparse Column format
     R = csc_matrix((data, (rows, cols)), shape=(hilbert_space_dim, hilbert_space_dim), dtype=np.float64)
     
     return R


def entanglement_entropy(psi: np.ndarray, num_qubits: int) -> float:
     """
     Calculates the von Neumann entanglement entropy of a subsystem.
     The system is partitioned into two equal halves.

     Args:
          psi (np.ndarray): The wavefunction (state vector) of the total system.
          num_qubits (int): The total number of qubits (L) in the system.

     Returns:
          float: The calculated entanglement entropy.
     """
          
     sub_system_size = num_qubits // 2
     
     # Normalize the wavefunction
     norm = np.linalg.norm(psi)
     if not np.isclose(norm, 1.0):
          psi = psi / norm
     
     # Reshape the state vector into a matrix representation.
     # The rows correspond to the Hilbert space of subsystem A,
     # and the columns to the Hilbert space of subsystem B.
     dim_A = 2**sub_system_size
     dim_B = 2**(num_qubits - sub_system_size)
     psi_matrix = psi.reshape((dim_A, dim_B))
     
     # Calculate the reduced density matrix for subsystem A by tracing out subsystem B.
     # rho_A = Tr_B(|psi><psi|) = psi_matrix @ psi_matrix.conj().T
     rho_A = np.dot(psi_matrix, psi_matrix.conj().T)
     
     # The eigenvalues of the reduced density matrix.
     # Since rho_A is Hermitian, eigvalsh is efficient and returns real eigenvalues.
     eigenvalues = np.linalg.eigvalsh(rho_A)
     
     # Filter out zero or negative eigenvalues to avoid issues with log.
     # Due to numerical precision, some eigenvalues might be very small and negative.
     positive_eigenvalues = eigenvalues[eigenvalues > 1e-12]
     
     # Calculate the von Neumann entropy: S = -Tr(rho_A * log(rho_A))
     # This is equivalent to -sum(lambda * log(lambda)) for eigenvalues lambda.
     with warnings.catch_warnings():
          warnings.simplefilter("ignore") # Suppress log(0) warnings if any slip through
          entropy = -np.sum(positive_eigenvalues * np.log(positive_eigenvalues))
          
     return entropy


def average_entanglement_entropy(initial_wavefunction: np.ndarray, num_qubits: int) -> float:
     """
     Calculates the entanglement entropy averaged over all L cyclic permutations
     of the initial wavefunction.

     Args:
          initial_wavefunction (np.ndarray): The starting state vector.
          num_qubits (int): The total number of qubits (L).

     Returns:
          float: The averaged entanglement entropy.
     """
     # Normalize the initial state
     norm = np.linalg.norm(initial_wavefunction)
     if not np.isclose(norm, 1.0):
          initial_wavefunction = initial_wavefunction / norm

     # Get the one-roll operator
     R = one_roll_operator(num_qubits)
     
     rolled_wavefunction = initial_wavefunction.copy()
     entropies = []
     
     # Calculate entropy for each of the L rolled states
     for _ in range(num_qubits):
          entropy = entanglement_entropy(rolled_wavefunction, num_qubits)
          entropies.append(entropy)
          # Apply the roll operator to get the next state
          rolled_wavefunction = R.dot(rolled_wavefunction)
          
     # Return the average of the calculated entropies
     return np.mean(entropies)


for L in L_lst:
     for k in k_lst:

          H_1_0 = load_npz(f'H_1_0_L_{L}_k_{k}_sigma_0.0.npz')
          H_0_1 = load_npz(f'k_local_hamiltonian_L_{L}_k_{k}.npz')

          def construct_H_mu_sigma(mu, sigma, H_0_1):
               return mu * H_1_0 + sigma * H_0_1

          # Construct the Hamiltonian for the given mu and sigma
          H_mu_sigma = construct_H_mu_sigma(mu, sigma, H_0_1)

          eigenvalues, eigenstates = eigh(H_mu_sigma.toarray())
          # Entanglement entropy for each eigenstate
          entanglement_entropy_lst = []
          for state in eigenstates.T:
               entropy = average_entanglement_entropy(state, L)
               entanglement_entropy_lst.append(entropy)
          # Save the eigenvalues, entanglement entropies using npz
          save_npz(f'eigenvalues_L_{L}_k_{k}_mu_{mu}_sigma_{sigma}.npz', csr_matrix(eigenvalues))
          save_npz(f'entanglement_entropy_L_{L}_k_{k}_mu_{mu}_sigma_{sigma}.npz', csr_matrix(entanglement_entropy_lst))