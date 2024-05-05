import numpy as np
from typing import List, Tuple, Union
from collections.abc import Iterable
import functools
import itertools
from math import log


QUBIT_PAULI_BASIS = {"I": np.array([[1, 0], [0, 1]]).astype(complex) / np.sqrt(2),
                     "X": np.array([[0, 1], [1, 0]]).astype(complex) / np.sqrt(2),
                     "Y": np.array([[0, -1j], [1j, 0]]).astype(complex) / np.sqrt(2),
                     "Z": np.array([[1, 0], [0, -1]]).astype(complex) / np.sqrt(2)}


def tensor(*args) -> np.ndarray:
    matrices = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            matrices.append(arg)
        elif isinstance(arg, Iterable):
            matrices.extend(arg)
        else:
            raise Exception("in tensor(): incorrect argument inputs for the tensor function!")

    return functools.reduce(lambda A, B: np.kron(A, B), matrices)


def partial_trace(A: np.ndarray, indices: List[int], n: int, dim: int) -> np.ndarray:
    if not indices:
        return A
    k = len(indices)
    indices.sort()
    qubit_axis = [(i, n + i) for i in indices]
    minus_factor = [(i, dim * i) for i in range(k)]
    minus_qubit_axis = [(q[0] - m[0], q[1] - m[1])
                        for q, m in zip(qubit_axis, minus_factor)]
    res = np.reshape(A, [dim, dim] * n)
    num_preserve = n - k
    for i, j in minus_qubit_axis:
        res = np.trace(res, axis1=i, axis2=j)
    if num_preserve > 1:
        res = np.reshape(res, [dim ** num_preserve] * 2)
    return res


def complete_pauli_matrices(n: int):
    pauli_names = itertools.product(list(QUBIT_PAULI_BASIS), repeat=n)

    pauli_matrices = []
    for name in pauli_names:
        pauli_operators = [QUBIT_PAULI_BASIS.get(ch) for ch in name]
        p_mat = tensor(pauli_operators)
        pauli_matrices.append(p_mat)

    return pauli_matrices


def operator_to_ptm(A: np.ndarray):
    n = int(np.log2(A.shape[0]))
    pauli_basis = complete_pauli_matrices(n)
    coes = []
    for pauli in pauli_basis:
        coe = np.trace(A @ pauli)
        coes.append(np.real(coe))

    return coes


def ptm_to_operator(coes: Union[List[float], np.ndarray]):
    if isinstance(coes, List):
        coes = np.asarray(coes)
    # Reshape to column vector
    coes = coes.reshape((coes.size, ))
    # Number of qubits
    n = int(log(coes.size, 4))
    pauli_basis = complete_pauli_matrices(n)
    op = np.zeros((2 ** n, 2 ** n))

    for i in range(coes.size):
        op = op + coes[i] * pauli_basis[i]

    return op



