import numpy as np
from itertools import combinations, product
from functools import reduce
from scipy.sparse import identity, csr_matrix, kron

class KLocalHamiltonian:
    def __init__(self, L, k, random_coefficients, adjacent=True):
        """
        Initialize the KLocalHamiltonian class.

        Parameters:
        - L: Number of qubits in the system.
        - k: Number of qubits involved in each k-local term.
        - random_coefficients: List of coefficients for the k-local terms.
        - adjacent: Whether the k-local terms are on adjacent qubits.
        """
        self.L = L
        self.k = k
        self.random_coefficients = random_coefficients
        self.adjacent = adjacent
        self.paulis = ['X', 'Y', 'Z']  # Pauli operators

        # Generate all k-local Pauli strings based on the configuration
        self.k_local_paulis = self.generate_k_local_pauli_strings() 

    def pauli_matrix(self, p):
        """
        Return the matrix representation of a given Pauli operator.

        Parameters:
        - p: A string representing the Pauli operator ('I', 'X', 'Y', 'Z').

        Returns:
        - A sparse matrix representation of the Pauli operator.
        """
        if p == "I":
            return identity(2, format="csr")  # Identity matrix
        elif p == "X":
            return csr_matrix([[0, 1], [1, 0]], dtype=complex)  # Pauli-X
        elif p == "Y":
            return csr_matrix([[0, -1j], [1j, 0]], dtype=complex)  # Pauli-Y
        elif p == "Z":
            return csr_matrix([[1, 0], [0, -1]], dtype=complex)  # Pauli-Z
        else:
            raise ValueError(f"Invalid Pauli operator: {p}")

    def tensor_product(self, pauli_string):
        """
        Compute the tensor product of a list of Pauli operators.

        Parameters:
        - pauli_string: A list of Pauli operators (e.g., ['X', 'I', 'Z']).

        Returns:
        - The tensor product as a sparse matrix.
        """
        matrices = [self.pauli_matrix(p) for p in pauli_string]
        return reduce(kron, matrices)  # Compute the tensor product

    def generate_k_local_pauli_strings(self):
        """
        Generate all possible k-local Pauli strings for the system.

        Returns:
        - A list of k-local Pauli strings.
        """
        pauli_strings = []

        # Generate adjacent k-local terms (without periodic boundary conditions)
        if self.adjacent:
            for i in range(self.L - self.k + 1):  # Iterate over valid starting positions
                for active_ops in product(self.paulis, repeat=self.k):  # All combinations of k Pauli operators
                    pauli_string = ['I'] * self.L  # Start with identity operators
                    for offset, P in enumerate(active_ops):
                        pauli_string[i + offset] = P  # Replace with active Pauli operators
                    pauli_strings.append("".join(pauli_string))

        # Generate non-adjacent k-local terms (k non-identity Pauli operators anywhere)
        if not self.adjacent:
            index_combinations = combinations(range(self.L), self.k)  # All combinations of k indices
            for indices in index_combinations:
                for active_ops in product(self.paulis, repeat=self.k):  # All combinations of k Pauli operators
                    pauli_string = ['I'] * self.L  # Start with identity operators
                    for idx, P in zip(indices, active_ops):
                        pauli_string[idx] = P  # Replace with active Pauli operators
                    pauli_strings.append("".join(pauli_string))                    

        return pauli_strings

    def hamiltonian_matrix(self):
        """
        Construct the Hamiltonian matrix for the system.

        Returns:
        - The Hamiltonian as a sparse matrix.
        """
        dim = 2 ** self.L  # Dimension of the Hilbert space
        total_matrix = csr_matrix((dim, dim), dtype=complex)  # Initialize the Hamiltonian matrix

        # Add each k-local term to the Hamiltonian
        for i, pauli_str in enumerate(self.k_local_paulis):
            total_matrix += self.random_coefficients[i] * self.tensor_product(pauli_str)

        return total_matrix