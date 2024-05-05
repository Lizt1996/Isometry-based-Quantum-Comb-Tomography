# An example of data collection of a process tensor.
# Process tensor is defined by a series of random isometries.


from data_collection_instrument_based import save_data, load_data, data_collect
import random
from ProcessTensorIsometry import InstProcTensorIsoTomography
from utils.la import partial_trace, complete_pauli_matrices
from utils.pt_isometry import random_isometry
import copy
import numpy as np

n_qubit = 1
n_time_step = 3

in_qlist = [n_qubit for _ in range(n_time_step)]
out_qlist = [n_qubit for _ in range(n_time_step)]
# Full ancillary dimensions
anc_qlist = None

# Specified ancillary dimensions
# anc_qlist = [1, 2]

random.seed(0)

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

    choi_state += np.kron(unitary @ s0 @ s0.conj().transpose() @ unitary.conj().transpose(),
                          s0 @ s0.conj().transpose())
    choi_state += np.kron(unitary @ s0 @ s1.conj().transpose() @ unitary.conj().transpose(),
                          s1 @ s0.conj().transpose())
    choi_state += np.kron(unitary @ s1 @ s1.conj().transpose() @ unitary.conj().transpose(),
                          s1 @ s1.conj().transpose())
    choi_state += np.kron(unitary @ s1 @ s0.conj().transpose() @ unitary.conj().transpose(),
                          s0 @ s1.conj().transpose())

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



# Collect measurement data
anc_qlist_from_A0 = [0] + [sum(in_qlist[:_t + 1]) for _t in range(n_time_step)]
test_isometries = [random_isometry(n_qubits_in=in_qlist[_t] + anc_qlist_from_A0[_t],
                                   n_qubits_out=out_qlist[_t] + anc_qlist_from_A0[_t + 1])
                   for _t in range(n_time_step)]

measurement_data = data_collect(test_isometries, complete_state_basis, cptp_choi_list, complete_povm_list)


iptt = InstProcTensorIsoTomography(complete_state_basis, cptp_choi_list, complete_povm_list, ancilla_qubits=anc_qlist)


# reconstruct quantum network via learning isometries on the Stiefel manifold
reconstructed_isometries = iptt.fit_stiefel(measurement_data,
                                            max_iterations=10000,
                                            print_cost=True, alpha=0.2, eps=1e-8, delta=1e-4, beta1=0.9,
                                            beta2=0.999, method='adam')

