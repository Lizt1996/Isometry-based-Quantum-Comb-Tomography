from utils.la import complete_pauli_matrices, operator_to_ptm, ptm_to_operator
from numba.core import types
from numba.typed import Dict, List
from numba import jit
from data_collection_for_compressive_ptt import *
import sys
from utils.pt_isometry import identity_isometry, isometries_to_choi_state


threshold = 1e-3

# tag = str(sys.argv[1])
# n_exps = int(sys.argv[2])
# piece_prefix = int(sys.argv[3])

tag = '11-11'
# n_exps = 10
# piece_prefix = 0

in_tag, out_tag = tag.split('-')

# set experiment qubits
in_qlist = tuple([int(in_tag[i]) for i in range(len(in_tag))])
out_qlist = tuple([int(out_tag[i]) for i in range(len(out_tag))])
n_time_step = len(in_qlist)


ancilla_qubits = [sum(out_qlist[:k + 1]) for k in range(n_time_step)]

isometry_input_qubits = [in_qlist[0]] + [in_qlist[i] + ancilla_qubits[i - 1] for i in range(1, n_time_step)]
isometry_output_qubits = [out_qlist[i] + ancilla_qubits[i] for i in range(n_time_step)]

id_isometries = [identity_isometry(isometry_input_qubits[_i], isometry_output_qubits[_i]) for _i in range(n_time_step)]

init_choi = isometries_to_choi_state(id_isometries, list(in_qlist), list(out_qlist))

n_gross_qubits = sum(in_qlist) + sum(out_qlist)

choi_dim = 2 ** n_gross_qubits

complete_in_qstate_list = []
complete_povm_list = []
for _t in range(n_time_step):
    complete_pb = complete_pauli_matrices(in_qlist[_t])
    # construct complete pauli basis
    complete_pauli_matrix = [np.sqrt(2) ** (in_qlist[_t]) * pmat for pmat in complete_pb]

    complete_state_basis = [1 / (2 ** in_qlist[_t]) * (complete_pauli_matrix[0] - complete_pauli_matrix[1])] + \
                           [1 / (2 ** in_qlist[_t]) * (complete_pauli_matrix[0] + pmat) for pmat in complete_pauli_matrix[1:]]

    complete_povm_basis = copy.deepcopy(complete_state_basis)

    complete_in_qstate_list.append(complete_state_basis)
    complete_povm_list.append(complete_povm_basis)

# # random isometries
# anc_qlist_from_A0 = [0] + [sum(in_qlist[:_t + 1]) for _t in range(n_time_step)]
# test_isometries = [random_isometry(n_qubits_in=in_qlist[_t] + anc_qlist_from_A0[_t],
#                                    n_qubits_out=out_qlist[_t] + anc_qlist_from_A0[_t + 1])
#                    for _t in range(n_time_step)]
#
# collected_data = data_collect(isometries=test_isometries,
#                               input_states=complete_in_qstate_list,
#                               povm_operators=complete_povm_list)

input_states = complete_in_qstate_list
povm_operators = complete_povm_list

input_indexes = []
povm_indexes = []

state_basis_size = [len(input_states[t]) for t in range(n_time_step)]
povm_basis_size = [len(povm_operators[t]) for t in range(n_time_step)]

for _t in range(n_time_step):
    input_indexes.append(list(
        itertools.product(*tuple(list(range(state_basis_size[_i])) for _i in range(_t + 1)))))

    povm_indexes.append(list(
        itertools.product(*tuple(list(range(povm_basis_size[_i])) for _i in range(_t + 1)))))

numba_input_indexes = []
for t in range(n_time_step):
    for index in input_indexes[t]:
        tmp_index = [-1 for _ in range(n_time_step)]
        for i in range(len(index)):
            tmp_index[i] = index[i]
        numba_input_indexes.append(tuple(tmp_index))
numba_input_indexes = tuple(numba_input_indexes)

# numba_input_indexes = np.array(numba_input_indexes)

numba_povm_indexes = []
for t in range(n_time_step):
    tmp_indexes = []
    for index in povm_indexes[t]:
        tmp_index = [-1 for _ in range(n_time_step)]
        for i in range(len(index)):
            tmp_index[i] = index[i]
        tmp_indexes.append(tuple(tmp_index))
    tmp_indexes = tuple(tmp_indexes)
    numba_povm_indexes.append(tmp_indexes)
numba_povm_indexes = tuple(numba_povm_indexes)

numba_input_states = []
for t in range(n_time_step):
    tmp_states = tuple(input_states[t])
    numba_input_states.append(tmp_states)
numba_input_states = tuple(numba_input_states)

numba_povm_operators = []
for t in range(n_time_step):
    tmp_states = tuple(povm_operators[t])
    numba_povm_operators.append(tmp_states)
numba_povm_operators = tuple(numba_povm_operators)


@jit(nopython=True)
def cost_fn_with_numba_N2(X, mdata: types.DictType(types.UniTuple(types.UniTuple(types.int64, 2), 2), types.float64)):
    # _Y = _get_choi_from_params(params)
    _cost = 0

    for i_0 in range(4 ** in_qlist[0]):
        for o_0 in range(4 ** out_qlist[0]):
            _input_state = 1 / np.sqrt(2) ** in_qlist[1] * \
                           np.kron(numba_input_states[0][i_0], np.identity(2 ** in_qlist[1]))

            _povm = 1 / np.sqrt(2) ** out_qlist[1] * \
                    np.kron(numba_povm_operators[0][o_0], np.identity(2 ** out_qlist[1]))

            _K = np.kron(_povm, _input_state.transpose())
            _prob = np.real(np.trace(X @ _K))

            data_key = ((i_0, -1), (o_0, -1))

            _cost += (_prob - mdata[data_key]) ** 2

    for i_0 in range(4 ** in_qlist[0]):
        for o_0 in range(4 ** out_qlist[0]):
            for i_1 in range(4 ** in_qlist[1]):
                for o_1 in range(4 ** out_qlist[1]):
                    _input_state = np.kron(numba_input_states[0][i_0], numba_input_states[1][i_1])

                    _povm = np.kron(numba_povm_operators[0][o_0], numba_povm_operators[1][o_1])

                    _K = np.kron(_povm, _input_state.transpose())
                    _prob = np.real(np.trace(X @ _K))

                    data_key = ((i_0, i_1), (o_0, o_1))

                    _cost += (_prob - mdata[data_key]) ** 2

    return _cost


@jit(nopython=True)
def cost_fn_with_numba_N3(X, mdata: types.DictType(types.UniTuple(types.UniTuple(types.int64, 2), 2), types.float64)):
    # _Y = _get_choi_from_params(params)
    _cost = 0

    for i_0 in range(4 ** in_qlist[0]):
        for o_0 in range(4 ** out_qlist[0]):
            _input_state = 1 / np.sqrt(2) ** in_qlist[1] * \
                           np.kron(numba_input_states[0][i_0], np.identity(2 ** in_qlist[1]))
            _input_state = 1 / np.sqrt(2) ** in_qlist[2] * \
                           np.kron(_input_state, np.identity(2 ** in_qlist[2]))

            _povm = 1 / np.sqrt(2) ** out_qlist[1] * \
                    np.kron(numba_povm_operators[0][o_0], np.identity(2 ** out_qlist[1]))
            _povm = 1 / np.sqrt(2) ** out_qlist[2] * \
                    np.kron(_povm, np.identity(2 ** out_qlist[2]))

            _K = np.kron(_povm, _input_state.transpose())
            _prob = np.real(np.trace(X @ _K))

            data_key = ((i_0, -1, -1), (o_0, -1, -1))

            _cost += (_prob - mdata[data_key]) ** 2

    for i_0 in range(4 ** in_qlist[0]):
        for o_0 in range(4 ** out_qlist[0]):
            for i_1 in range(4 ** in_qlist[1]):
                for o_1 in range(4 ** out_qlist[1]):
                    _input_state = np.kron(numba_input_states[0][i_0], numba_input_states[1][i_1])
                    _input_state = 1 / np.sqrt(2) ** in_qlist[2] * \
                                   np.kron(_input_state, np.identity(2 ** in_qlist[2]))

                    _povm = np.kron(numba_povm_operators[0][o_0], numba_povm_operators[1][o_1])
                    _povm = 1 / np.sqrt(2) ** out_qlist[2] * \
                            np.kron(_povm, np.identity(2 ** out_qlist[2]))

                    _K = np.kron(_povm, _input_state.transpose())
                    _prob = np.real(np.trace(X @ _K))

                    data_key = ((i_0, i_1, -1), (o_0, o_1, -1))

                    _cost += (_prob - mdata[data_key]) ** 2

    for i_0 in range(4 ** in_qlist[0]):
        for o_0 in range(4 ** out_qlist[0]):
            for i_1 in range(4 ** in_qlist[1]):
                for o_1 in range(4 ** out_qlist[1]):
                    for i_2 in range(4 ** in_qlist[2]):
                        for o_2 in range(4 ** out_qlist[2]):
                            _input_state = np.kron(numba_input_states[0][i_0], numba_input_states[1][i_1])
                            _input_state = np.kron(_input_state, numba_input_states[2][i_2])

                            _povm = np.kron(numba_povm_operators[0][o_0], numba_povm_operators[1][o_1])
                            _povm = np.kron(_povm, numba_povm_operators[2][o_2])

                            _K = np.kron(_povm, _input_state.transpose())
                            _prob = np.real(np.trace(X @ _K))

                            data_key = ((i_0, i_1, i_2), (o_0, o_1, o_2))

                            _cost += (_prob - mdata[data_key]) ** 2

    return _cost


@jit(nopython=True)
def grad_fn_with_numba_N2(X, mdata):
    # _Y = _get_choi_from_params(params)
    _grad = np.empty((choi_dim, choi_dim), dtype=types.complex128)

    for i_0 in range(4 ** in_qlist[0]):
        for o_0 in range(4 ** out_qlist[0]):
            _input_state = 1 / np.sqrt(2) ** in_qlist[1] * \
                           np.kron(numba_input_states[0][i_0], np.identity(2 ** in_qlist[1]))

            _povm = 1 / np.sqrt(2) ** out_qlist[1] * \
                    np.kron(numba_povm_operators[0][o_0], np.identity(2 ** out_qlist[1]))

            _K = np.kron(_povm, _input_state.transpose())

            _prob = np.real(np.trace(X @ _K))

            data_key = ((i_0, -1), (o_0, -1))

            _grad += (_prob - mdata[data_key]) * _K.conj().transpose()

    for i_0 in range(4 ** in_qlist[0]):
        for o_0 in range(4 ** out_qlist[0]):
            for i_1 in range(4 ** in_qlist[1]):
                for o_1 in range(4 ** out_qlist[1]):
                    _input_state = np.kron(numba_input_states[0][i_0], numba_input_states[1][i_1])

                    _povm = np.kron(numba_povm_operators[0][o_0], numba_povm_operators[1][o_1])

                    _K = np.kron(_povm, _input_state.transpose())
                    _prob = np.real(np.trace(X @ _K))

                    data_key = ((i_0, i_1), (o_0, o_1))

                    _grad += (_prob - mdata[data_key]) * _K.conj().transpose()

    return _grad

@jit(nopython=True)
def grad_fn_with_numba_N3(X, mdata):
    # _Y = _get_choi_from_params(params)
    _grad = np.empty((choi_dim, choi_dim), dtype=types.complex128)

    for i_0 in range(4 ** in_qlist[0]):
        for o_0 in range(4 ** out_qlist[0]):
            _input_state = 1 / np.sqrt(2) ** in_qlist[1] * \
                           np.kron(numba_input_states[0][i_0], np.identity(2 ** in_qlist[1]))
            _input_state = 1 / np.sqrt(2) ** in_qlist[2] * \
                           np.kron(_input_state, np.identity(2 ** in_qlist[2]))

            _povm = 1 / np.sqrt(2) ** out_qlist[1] * \
                    np.kron(numba_povm_operators[0][o_0], np.identity(2 ** out_qlist[1]))
            _povm = 1 / np.sqrt(2) ** out_qlist[2] * \
                    np.kron(_povm, np.identity(2 ** out_qlist[2]))

            _K = np.kron(_povm, _input_state.transpose())

            _prob = np.real(np.trace(X @ _K))

            data_key = ((i_0, -1, -1), (o_0, -1, -1))

            _grad += (_prob - mdata[data_key]) * _K.conj().transpose()

    for i_0 in range(4 ** in_qlist[0]):
        for o_0 in range(4 ** out_qlist[0]):
            for i_1 in range(4 ** in_qlist[1]):
                for o_1 in range(4 ** out_qlist[1]):
                    _input_state = np.kron(numba_input_states[0][i_0], numba_input_states[1][i_1])
                    _input_state = 1 / np.sqrt(2) ** in_qlist[2] * \
                                   np.kron(_input_state, np.identity(2 ** in_qlist[2]))

                    _povm = np.kron(numba_povm_operators[0][o_0], numba_povm_operators[1][o_1])
                    _povm = 1 / np.sqrt(2) ** out_qlist[2] * \
                            np.kron(_povm, np.identity(2 ** out_qlist[2]))

                    _K = np.kron(_povm, _input_state.transpose())
                    _prob = np.real(np.trace(X @ _K))

                    data_key = ((i_0, i_1, -1), (o_0, o_1, -1))

                    _grad += (_prob - mdata[data_key]) * _K.conj().transpose()


    for i_0 in range(4 ** in_qlist[0]):
        for o_0 in range(4 ** out_qlist[0]):
            for i_1 in range(4 ** in_qlist[1]):
                for o_1 in range(4 ** out_qlist[1]):
                    for i_2 in range(4 ** in_qlist[2]):
                        for o_2 in range(4 ** out_qlist[2]):
                            _input_state = np.kron(numba_input_states[0][i_0], numba_input_states[1][i_1])
                            _input_state = np.kron(_input_state, numba_input_states[2][i_2])

                            _povm = np.kron(numba_povm_operators[0][o_0], numba_povm_operators[1][o_1])
                            _povm = np.kron(_povm, numba_povm_operators[2][o_2])

                            _K = np.kron(_povm, _input_state.transpose())
                            _prob = np.real(np.trace(X @ _K))

                            data_key = ((i_0, i_1, i_2), (o_0, o_1, o_2))

                            _grad += (_prob - mdata[data_key]) * _K.conj().transpose()

    return _grad



def _get_zero_indexes():
    _ptm_dim = [4 ** q for q in out_qlist] + [4 ** q for q in in_qlist]
    _ptm_dim.reverse()
    zero_index = []

    for _time_step in range(n_time_step):
        out_list_of_iter_meta = ([list(range(4 ** out_qlist[i])) for i in range(_time_step)] +
                                 [[0] for _ in range(n_time_step - _time_step)])
        in_list_of_iter_meta = ([list(range(4 ** in_qlist[i])) for i in range(_time_step)] +
                                [list(range(1, 4 ** in_qlist[_time_step]))] +
                                [[0] for _ in range(n_time_step - _time_step - 1)])

        list_of_iter_meta = out_list_of_iter_meta + in_list_of_iter_meta
        iter_indexes = list(itertools.product(*tuple(list_of_iter_meta)))
        for indexes in iter_indexes:
            lst_index = list(indexes)
            lst_index.reverse()
            index = 0
            for i in range(len(lst_index)):
                index += lst_index[i] * np.prod(_ptm_dim[:i], dtype=int)
                k = 1

            zero_index.append(index)
    zero_index.sort()
    return zero_index


def _get_causality_const_mat():
    _zero_indexes = _get_zero_indexes()
    _gross_ptm_dims = 4 ** sum(in_qlist + out_qlist)

    const_list = []
    id_vec = [0. for _ in range(_gross_ptm_dims)]
    id_vec[0] = 1.
    const_list.append(id_vec)

    for i in _zero_indexes:
        p_vec = [0. for _ in range(_gross_ptm_dims)]
        p_vec[i] = 1
        const_list.append(p_vec)

    b_size = len(const_list)
    b_vec = [0. for _ in range(b_size)]
    b_vec[0] = 1.

    return np.array(const_list), np.array(b_vec).reshape((b_size, 1)), len(const_list)


_causality_mat, _causality_vec, _causality_const_num = _get_causality_const_mat()


def _cp_project(sop_Y):
    _Y = ptm_to_operator(sop_Y)
    _val, _vec = np.linalg.eig(_Y)
    for _i in range(len(_val)):
        _val[_i] = _val[_i] if _val[_i] > 0 else 0
    return np.array(operator_to_ptm(_vec @ np.diag(_val) @ _vec.conj().transpose())).reshape(sop_Y.shape)


def _causality_project(sop_Y):
    core = np.linalg.inv(_causality_mat @ _causality_mat.conj().transpose())

    _proj_v = np.identity(sop_Y.shape[0]) - _causality_mat.conj().transpose() @ core @ _causality_mat
    _proj_b = _causality_mat.conj().transpose() @ core

    projected_sop_Y = _proj_v @ sop_Y + _proj_b @ _causality_vec
    return projected_sop_Y


# Dikstra
def _dikstra_project(_Y):
    sop_Y = operator_to_ptm(_Y)
    sop_Y = np.array(sop_Y).reshape((len(sop_Y), 1))

    _x = sop_Y
    _y = np.zeros(sop_Y.shape)
    _p = np.zeros(sop_Y.shape)
    _q = np.zeros(sop_Y.shape)

    while True:
        _x_prime = _x
        _y_prime = _y
        _p_prime = _p
        _q_prime = _q

        _y = _causality_project(_x + _p)
        _p = _x + _p - _y
        _x = _cp_project(_y + _q)
        _q = _y + _q - _x

        if np.linalg.norm(_p_prime - _p) ** 2 + np.linalg.norm(_q_prime - _q) ** 2 + abs(
                2 * _p_prime.transpose() @ (_x - _x_prime)) + abs(
            2 * _q_prime.transpose() @ (_y - _y_prime)) <= threshold:
            break
    return ptm_to_operator(_x)


def fit(measurement_data, **kwargs):
    grad_fn = None
    cost_fn = None
    if len(in_qlist) == 2:
        grad_fn = grad_fn_with_numba_N2
        cost_fn = cost_fn_with_numba_N2
    elif len(in_qlist) == 3:
        pass
    else:
        exit(0)

    threshold = kwargs.get('threshold', 1e-4)
    tol = kwargs.get('tol', 1e-3)
    dtol = kwargs.get('dtol', 1e-6)
    gamma = kwargs.get('gamma', 0.3)
    max_alpha_dec = kwargs.get('max_alpha_dec', 1000)
    beta0 = kwargs.get('beta0', 0.05)

    numba_mea_data = Dict.empty(
        key_type=types.UniTuple(types.UniTuple(types.int64, n_time_step), 2),
        value_type=types.float64,
    )

    for t in range(n_time_step):
        for key in measurement_data[t]:
            input_key = key[0]
            output_key = key[1]

            tmp_input_key = [-1 for _ in range(n_time_step)]
            tmp_output_key = [-1 for _ in range(n_time_step)]
            for i in range(t + 1):
                tmp_input_key[i] = input_key[i]
                tmp_output_key[i] = output_key[i]
            tmp_input_key = tuple(tmp_input_key)
            tmp_output_key = tuple(tmp_output_key)
            expanded_key = (tmp_input_key, tmp_output_key)

            numba_mea_data[expanded_key] = measurement_data[t][key]

    # ------------------ main optimization process-----------------

    _Y = init_choi

    G = grad_fn(_Y, mdata=numba_mea_data)
    cost_val = cost_fn(_Y, mdata=numba_mea_data)

    alpha = 3 / (2 * _Y.shape[0] ** 2)
    last_cost_val = cost_val + 100000
    while np.linalg.norm(G) > tol:
        cost_val = cost_fn(_Y, mdata=numba_mea_data)
        if last_cost_val - cost_val < 1e-6:
            break

        G = grad_fn(_Y, mdata=numba_mea_data)
        D = _dikstra_project(_Y - alpha * G) - _Y
        beta = 1

        dir_innerprod = np.trace(D.conj().transpose() @ G)
        _counter = 0
        while cost_fn(_Y + beta * D, mdata=numba_mea_data) > cost_val + beta * gamma * dir_innerprod:
            beta = 0.5 * beta
            _counter += 1
            if _counter >= max_alpha_dec:
                break
            if beta < beta0:
                beta = beta0
                break

        _Y = _Y + beta * D

        # new_cost_val = _cost_fn(Y)
        # if abs(new_cost_val - cost_val) < tol:
        #     break
        # cost_val = new_cost_val
        last_cost_val = cost_val
        print(cost_fn(_Y, mdata=numba_mea_data))

    return _Y, cost_fn(_Y, mdata=numba_mea_data)




if __name__ == '__main__':

    prob_path = './test_prob.prob'
    collected_data = load_data(prob_path)

    numba_mea_data = Dict.empty(
        key_type=types.UniTuple(types.UniTuple(types.int64, n_time_step), 2),
        value_type=types.float64,
    )

    for t in range(n_time_step):
        for key in collected_data[t]:
            input_key = key[0]
            output_key = key[1]

            tmp_input_key = [-1 for _ in range(n_time_step)]
            tmp_output_key = [-1 for _ in range(n_time_step)]
            for i in range(t + 1):
                tmp_input_key[i] = input_key[i]
                tmp_output_key[i] = output_key[i]
            tmp_input_key = tuple(tmp_input_key)
            tmp_output_key = tuple(tmp_output_key)
            expanded_key = (tmp_input_key, tmp_output_key)

            numba_mea_data[expanded_key] = collected_data[t][key]

    choi, _ = fit(collected_data, threshold=1e-3, tol=1e-3, dtol=1e-6, gamma=0.3)


