
from utils.la import complete_pauli_matrices
from data_collection_for_compressive_ptt import save_data, load_data, data_collect
import random
from ProcessTensorIsometry import ProcTensorIsometryTomography
from utils.pt_isometry import random_isometry, isometries_to_choi_state
import copy
import numpy as np

in_qlist = [1, 1]
out_qlist = [1, 1]


# Full ancillary dimensions
anc_qlist = None

# Specified ancillary dimensions
# anc_qlist = [1, 2]

random.seed(0)

n_time_step = len(in_qlist)

# Full ancillary dimensions for data collection
anc_qlist_from_A0 = [0] + [sum(in_qlist[:_t + 1]) for _t in range(n_time_step)]
test_isometries = [random_isometry(n_qubits_in=in_qlist[_t] + anc_qlist_from_A0[_t],
                                   n_qubits_out=out_qlist[_t] + anc_qlist_from_A0[_t + 1])
                   for _t in range(n_time_step)]

ideal_choi_state = isometries_to_choi_state(test_isometries,
                                            input_qubits=in_qlist,
                                            output_qubits=out_qlist)

complete_in_qstate_list = []
complete_povm_list = []
for _t in range(n_time_step):
    complete_pb = complete_pauli_matrices(in_qlist[_t])
    # construct complete pauli basis
    complete_pauli_matrix = [np.sqrt(2) ** (in_qlist[_t]) * pmat for pmat in complete_pb]

    complete_state_basis = [1 / 2 * (complete_pauli_matrix[0] - complete_pauli_matrix[1])] + \
                           [1 / 2 * (complete_pauli_matrix[0] + pmat) for pmat in complete_pauli_matrix[1:]]

    complete_povm_basis = copy.deepcopy(complete_state_basis)

    complete_in_qstate_list.append(complete_state_basis)
    complete_povm_list.append(complete_povm_basis)

# Collect measurement data
collected_data = data_collect(isometries=test_isometries,
                              input_states=complete_in_qstate_list,
                              povm_operators=complete_povm_list)


ptt = ProcTensorIsometryTomography(input_states=complete_in_qstate_list,
                                   povm_operators=complete_povm_list,
                                   ancilla_qubits=anc_qlist)

# reconstruct quantum network via learning isometries on the Stiefel manifold
reconstructed_isometries, _ = ptt.fit_stiefel(collected_data,
                                              max_iterations=10000,
                                              print_cost=True, alpha=0.1, eps=1e-8, delta=1e-4, beta1=0.9,
                                              beta2=0.999, method='adam')
