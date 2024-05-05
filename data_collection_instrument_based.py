import copy
import itertools

import numpy as np
from typing import List
import pickle
from utils.pt_isometry import get_out_state_with_instruments
from utils.la import partial_trace, complete_pauli_matrices


from tqdm import tqdm

n_qubit = 1
n_time_step = 3

# define cptp instruments

cptp_op_names = ['id', 'rz_90', 'rz_45', 'x', 'rz_90_x', 'rz_45_x', 'sx', 'rz_90_sx', 'rz_45_sx', 'id_x_cx',
                 'id_x_swap', 'id_sx_swap_cz']

unitary_x = np.array([[0, 1], [1, 0]], dtype=complex)
unitary_sx = 1 / 2 * np.array([[1. + 1j, 1. - 1j], [1. - 1j, 1. + 1j]], dtype=complex)
unitary_id = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)
unitary_rz_90 = np.array([[0.7071 - 0.7071j, 0.0000 + 0.0000j],
                          [0.0000 + 0.0000j, 0.7071 + 0.7071j]], dtype=complex)
unitary_rz_45 = np.array([[0.9239 - 0.3827j, 0.0000 + 0.0000j],
                          [0.0000 + 0.0000j, 0.9239 + 0.3827j]], dtype=complex)

unitary_cx = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]], dtype=complex)

unitary_cz = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, -1]], dtype=complex)

unitary_swap = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]], dtype=complex)

unitary_iswap = np.array([[1, 0, 0, 0],
                          [0, 0, 1j, 0],
                          [0, 1j, 0, 0],
                          [0, 0, 0, 1]], dtype=complex)


def single_qubit_unitary_to_choi(unitary: np.ndarray):
    s0 = np.array([[1],
                   [0]], dtype=complex)
    s1 = np.array([[0],
                   [1]], dtype=complex)

    choi_state = np.zeros((4, 4), dtype=complex)

    choi_state += np.kron(unitary @ s0 @ s0.conj().transpose() @ unitary.conj().transpose(), s0 @ s0.conj().transpose())
    choi_state += np.kron(unitary @ s0 @ s1.conj().transpose() @ unitary.conj().transpose(), s1 @ s0.conj().transpose())
    choi_state += np.kron(unitary @ s1 @ s1.conj().transpose() @ unitary.conj().transpose(), s1 @ s1.conj().transpose())
    choi_state += np.kron(unitary @ s1 @ s0.conj().transpose() @ unitary.conj().transpose(), s0 @ s1.conj().transpose())

    return choi_state


def cptp_by_two_qubit_unitary_with_0anc_to_choi(unitary: np.ndarray):
    s0 = np.array([[1],
                   [0]], dtype=complex)
    s1 = np.array([[0],
                   [1]], dtype=complex)

    choi_state = np.zeros((8, 8), dtype=complex)

    choi_state += np.kron(
        unitary @ np.kron(s0 @ s0.conj().transpose(), s0 @ s0.conj().transpose()) @ unitary.conj().transpose(),
        s0 @ s0.conj().transpose())
    choi_state += np.kron(
        unitary @ np.kron(s0 @ s1.conj().transpose(), s0 @ s0.conj().transpose()) @ unitary.conj().transpose(),
        s1 @ s0.conj().transpose())
    choi_state += np.kron(
        unitary @ np.kron(s1 @ s1.conj().transpose(), s0 @ s0.conj().transpose()) @ unitary.conj().transpose(),
        s1 @ s1.conj().transpose())
    choi_state += np.kron(
        unitary @ np.kron(s1 @ s0.conj().transpose(), s0 @ s0.conj().transpose()) @ unitary.conj().transpose(),
        s0 @ s1.conj().transpose())

    choi_state = partial_trace(choi_state, [1], 3, 2)

    return choi_state


sq_unitaries = [unitary_id, unitary_rz_90, unitary_rz_45,
                unitary_x, unitary_x @ unitary_rz_90, unitary_x @ unitary_rz_45,
                unitary_sx, unitary_sx @ unitary_rz_90, unitary_sx @ unitary_rz_45]

dq_unitaries = [unitary_cx @ np.kron(unitary_id, unitary_x),
                unitary_swap @ np.kron(unitary_id, unitary_x),
                unitary_cz @ unitary_swap @ unitary_cx @ np.kron(unitary_id, unitary_sx)]

sq_choi_states = [single_qubit_unitary_to_choi(su) for su in sq_unitaries]
# choi_states = sq_choi_states
dq_choi_states = [cptp_by_two_qubit_unitary_with_0anc_to_choi(du) for du in dq_unitaries]

cptp_choi_states = sq_choi_states + dq_choi_states

# define init state and povm basis
complete_pb = complete_pauli_matrices(n_qubit)
# construct complete pauli basis
complete_pauli_matrix = [np.sqrt(2) ** n_qubit * pmat for pmat in complete_pb]

complete_state_basis = [1 / 2 * (complete_pauli_matrix[0] - complete_pauli_matrix[1])] + \
                       [1 / 2 * (complete_pauli_matrix[0] + pmat) for pmat in complete_pauli_matrix[1:]]

complete_povm_basis = copy.deepcopy(complete_state_basis)

complete_povm_list = [complete_povm_basis for _ in range(n_time_step)]

cptp_choi_list = [cptp_choi_states for _ in range(n_time_step - 1)]


def data_collect(isometries: List[np.ndarray], init_states: List[np.ndarray],
                 cptp_operators: List[List[np.ndarray]], povm_operators: List[List[np.ndarray]]):
    """
    Collect complete probability data with informationally complete input states and POVM operators.
    :param isometries: the isometry represented process tensor. A list of isometries (np.ndarray).
    :param input_states: a list of informationally complete input states.
    :param povm_operators: a list of informationally complete POVM operators.
    :return: probability data, with structure
                [data_with_max_time_step_0, data_with_max_time_step_1, ..., data_with_max_time_step_N-1]
                data_with_max_time_step_k: [(input_state_index, output_state_index): prob...]
    """
    n_time_steps = len(isometries)
    dim = 4 ** n_qubit

    pieces = [dim ** (2 * (i + 1)) for i in range(n_time_step)]

    total_pieces = sum(pieces)
    pbar = tqdm(
        total=total_pieces,
        delay=3
    )

    collected_data = []
    for k in range(n_time_steps):
        init_state_indexes = list(range(4 ** n_qubit))

        cptp_indexes = list(itertools.product(*tuple(list(range((4 ** n_qubit) * (4 ** n_qubit - 1))) for _i in range(k))))

        povm_op_indexes = list(range(4 ** n_qubit))

        single_k_data = {}
        for init_state_index in init_state_indexes:
            for cptp_index in cptp_indexes:
                for povm_op_index in povm_op_indexes:

                    # Compute probabilities
                    init_state = init_states[init_state_index]
                    cptp_chois = [cptp_operators[t][cptp_index[t]] for t in range(k)]

                    out_state = get_out_state_with_instruments(isometries[:k + 1], init_state, cptp_chois)

                    povm_op = povm_operators[k][povm_op_index]
                    prob = np.real(np.trace(povm_op @ out_state))
                    # --------------------
                    single_k_data[(init_state_index, cptp_index, povm_op_index)] = prob
                    pbar.update(1)

        collected_data.append(single_k_data)

    pbar.close()

    return collected_data


def save_data(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_data(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

