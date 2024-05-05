import numpy as np
from typing import List, Tuple, Union
import itertools
import utils.la as la
import copy


def random_isometry(n_qubits_in: int, n_qubits_out: int, seed=None) -> np.ndarray:
    if n_qubits_out < n_qubits_in:
        raise ValueError('random_isometry: invalid intput of input and output qubits. The number of output qubits '
                         'must be equal to or more than it of input qubits.')
    _n_cols = 2 ** n_qubits_in
    _n_rows = 2 ** n_qubits_out
    _shape = (_n_rows, _n_cols)
    n_eigvals = min(_n_rows, _n_cols)

    if seed is not None:
        np.random.seed(seed)

    _A = np.random.uniform(-1, 1, _shape) + 1j * np.random.uniform(-1, 1, _shape)
    _unitary_out, _, _unitary_in = np.linalg.svd(_A)

    _eigvals = [np.exp(1j * np.random.uniform(-np.pi, np.pi)) for _ in range(n_eigvals)]

    _sigma = np.zeros(_shape, dtype=complex)
    for i in range(n_eigvals):
        _sigma[i][i] = _eigvals[i]

    _isometry = _unitary_out @ _sigma @ _unitary_in

    return _isometry


def identity_isometry(n_qubits_in: int, n_qubits_out: int) -> np.ndarray:
    if n_qubits_out < n_qubits_in:
        raise ValueError('random_isometry: invalid intput of input and output qubits. The number of output qubits '
                         'must be equal to or more than it of input qubits.')
    _n_cols = 2 ** n_qubits_in
    _n_rows = 2 ** n_qubits_out
    _shape = (_n_rows, _n_cols)

    id_iso = np.zeros(_shape, dtype=complex)

    for _i in range(min(_shape)):
        id_iso[_i][_i] = 1. + 0.j

    return id_iso


def random_state(n_qubits: int, seed=None) -> np.ndarray:
    _dim = 2 ** n_qubits
    _shape = (_dim, _dim)

    if seed is not None:
        np.random.seed(seed)

    _A = np.random.uniform(-1, 1, _shape) + 1j * np.random.uniform(-1, 1, _shape)
    rho = (_A @ _A.conj().transpose())
    U, sigma, V = np.linalg.svd(rho)
    sigma = sigma / sum(sigma)
    rho = U @ np.diag(sigma) @ V

    return rho


def get_entry_index(subsystem_indexes: Union[List[int], Tuple[int]], subsystem_qubits: Union[List[int], Tuple[int]]):
    r"""
    Get the index of 1-valued entry on the computational basis.
    The length of subsystem_indexes and subsystem_qubits should be the same.

    .. math::
        \vert s \rangle := \vert s_0 \rangle \otimes \dots \otimes \vert s_n \rangle


    :param subsystem_indexes: a list of index indicating the computational basis of subsystems.
    :param subsystem_qubits: a list of numbers of qubits of subsystems.
    :return: the index s of computational basis of the whole system.
    """
    s = 0
    for s_i, n_qubits in zip(subsystem_indexes, subsystem_qubits):
        s = s * 2 ** n_qubits + s_i
    return s


def get_ket(computational_basis: int, n_qubits: int) -> np.ndarray:
    ket = np.zeros((2 ** n_qubits, 1), dtype=complex)
    ket[computational_basis] = 1
    return ket


def fill_isometry(isometry: np.ndarray, in_qlist: List[int], out_qlist: List[int],
                  filling_n_qubit_list: List[int]) -> np.ndarray:
    r"""
    Fill the isometry by identities.

    :param isometry: The isometry to be filled;
    :param in_qlist: Indexes of input qubits indicating the starting dimension to perform the identity fillings;
    :param out_qlist: Indexes of output qubits indicating the starting dimension to perform the identity fillings;
    :param filling_n_qubit_list: A list of numbers of qubits which indicates the dimensions of filled identities;
    :return: filled isometry
    """
    V = copy.deepcopy(isometry)
    n_input_qubits = round(np.log2(V.shape[1]))
    n_output_qubits = round(np.log2(V.shape[0]))

    # input validation check
    if len(in_qlist) != len(out_qlist) or len(in_qlist) != len(filling_n_qubit_list):
        raise ValueError('Invalid input: sizes of in_qlist, out_qlist, and filling_n_qubit_list should be identical.')

    if max(in_qlist) > n_input_qubits or min(in_qlist) < 0 or len(set(in_qlist)) != len(in_qlist):
        raise ValueError('Invalid input: in_qlist.')

    if max(out_qlist) > n_output_qubits or min(out_qlist) < 0 or len(set(out_qlist)) != len(out_qlist):
        raise ValueError('Invalid input: out_qlist.')

    _in_qlist = []
    _out_qlist = []
    _filling_n_qubit_list = []

    for i in range(len(filling_n_qubit_list)):
        if filling_n_qubit_list[i] > 0:
            _in_qlist.append(in_qlist[i])
            _out_qlist.append(out_qlist[i])
            _filling_n_qubit_list.append(filling_n_qubit_list[i])

    # number of qubits of subsystems
    in_subsystem_n_qubit_list = [_in_qlist[0]] + \
                                [_in_qlist[i] - _in_qlist[i - 1] for i in range(1, len(_in_qlist))] + \
                                [n_input_qubits - _in_qlist[-1]]

    out_subsystem_n_qubit_list = [_out_qlist[0]] + \
                                 [_out_qlist[i] - _out_qlist[i - 1] for i in range(1, len(_out_qlist))] + \
                                 [n_output_qubits - _out_qlist[-1]]

    start_with_identity = (in_subsystem_n_qubit_list[0] == 0, out_subsystem_n_qubit_list[0] == 0)
    end_with_identity = (in_subsystem_n_qubit_list[-1] == 0, out_subsystem_n_qubit_list[-1] == 0)

    if start_with_identity[0]:
        in_subsystem_n_qubit_list.pop(0)
    if start_with_identity[1]:
        out_subsystem_n_qubit_list.pop(0)
    if end_with_identity[0]:
        in_subsystem_n_qubit_list.pop(-1)
    if end_with_identity[1]:
        out_subsystem_n_qubit_list.pop(-1)

    # total number of input qubits in the filled isometry
    n_filled_in_qubits = n_input_qubits + sum(_filling_n_qubit_list)

    # total number of output qubits in the filled isometry
    n_filled_out_qubits = n_output_qubits + sum(_filling_n_qubit_list)

    # initialization of output filled isometry
    V_filled = np.zeros((2 ** n_filled_out_qubits, 2 ** n_filled_in_qubits), dtype=complex)

    origin_isometry_in_indexes = list(itertools.product(*tuple(list(range(2 ** q)) for q in in_subsystem_n_qubit_list)))
    origin_isometry_out_indexes = list(
        itertools.product(*tuple(list(range(2 ** q)) for q in out_subsystem_n_qubit_list)))

    identity_indexes = list(itertools.product(*tuple(list(range(2 ** q)) for q in _filling_n_qubit_list)))

    for V_origin_in_index in origin_isometry_in_indexes:
        for V_origin_out_index in origin_isometry_out_indexes:
            for id_index in identity_indexes:

                V_origin_in_basis = get_entry_index(V_origin_in_index, in_subsystem_n_qubit_list)
                V_origin_out_basis = get_entry_index(V_origin_out_index, out_subsystem_n_qubit_list)

                filled_in_index = [id_index[0]] if start_with_identity[0] else []
                filled_in_subsystem_qubits = [_filling_n_qubit_list[0]] if start_with_identity[0] else []
                id_in_start_index = 1 if start_with_identity[0] else 0

                for i_of_index in range(len(V_origin_in_index)):
                    filled_in_index.append(V_origin_in_index[i_of_index])
                    filled_in_subsystem_qubits.append(in_subsystem_n_qubit_list[i_of_index])
                    if id_in_start_index + i_of_index < len(id_index):
                        filled_in_index.append(id_index[id_in_start_index + i_of_index])
                        filled_in_subsystem_qubits.append(_filling_n_qubit_list[id_in_start_index + i_of_index])

                V_filled_in_basis = get_entry_index(filled_in_index, filled_in_subsystem_qubits)

                filled_out_index = [id_index[0]] if start_with_identity[1] else []
                filled_out_subsystem_qubits = [_filling_n_qubit_list[0]] if start_with_identity[1] else []
                id_out_start_index = 1 if start_with_identity[1] else 0

                for i_of_index in range(len(V_origin_out_index)):
                    filled_out_index.append(V_origin_out_index[i_of_index])
                    filled_out_subsystem_qubits.append(out_subsystem_n_qubit_list[i_of_index])
                    if id_out_start_index + i_of_index < len(id_index):
                        filled_out_index.append(id_index[id_out_start_index + i_of_index])
                        filled_out_subsystem_qubits.append(_filling_n_qubit_list[id_out_start_index + i_of_index])

                V_filled_out_basis = get_entry_index(filled_out_index, filled_out_subsystem_qubits)

                ket_in = get_ket(V_filled_in_basis, n_filled_in_qubits)
                ket_out = get_ket(V_filled_out_basis, n_filled_out_qubits)

                V_filled += V[V_origin_out_basis][V_origin_in_basis] * ket_out @ ket_in.transpose().conj()

    return V_filled


def insert_operator(operator: np.ndarray, qlist: List[int], fillings: List[np.ndarray]) -> np.ndarray:
    rho = copy.deepcopy(operator)
    n_qubits = round(np.log2(rho.shape[0]))

    fillings_n_qubits = [round(np.log2(filling.shape[0])) for filling in fillings]

    subsystem_n_qubit_list = [qlist[0]] + \
                             [qlist[i] - qlist[i - 1] for i in range(1, len(qlist))] + \
                             [n_qubits - qlist[-1]]

    start_with_filling = (subsystem_n_qubit_list[0] == 0)
    end_with_filling = (subsystem_n_qubit_list[-1] == 0)

    if start_with_filling:
        subsystem_n_qubit_list.pop(0)
    if end_with_filling:
        subsystem_n_qubit_list.pop(-1)

    # total number of output qubits in the filled isometry
    output_qubits = n_qubits + sum(fillings_n_qubits)

    # initialization of output filled isometry
    eta = np.zeros((2 ** output_qubits, 2 ** output_qubits), dtype=complex)

    origin_operator_indexes = list(itertools.product(*tuple(list(range(2 ** q)) for q in subsystem_n_qubit_list)))
    fillings_indexes = list(itertools.product(*tuple(list(range(2 ** q)) for q in fillings_n_qubits)))

    for origin_ket_index in origin_operator_indexes:
        for origin_bra_index in origin_operator_indexes:
            for fillings_ket_index in fillings_indexes:
                for fillings_bra_index in fillings_indexes:
                    rho_ket_basis = get_entry_index(origin_ket_index, subsystem_n_qubit_list)
                    rho_bra_basis = get_entry_index(origin_bra_index, subsystem_n_qubit_list)

                    fillings_coe = [fillings[i][fillings_ket_index[i]][fillings_bra_index[i]]
                                    for i in range(len(fillings))]

                    coe = rho[rho_ket_basis][rho_bra_basis] * np.prod(fillings_coe)

                    inserted_ket_index = [fillings_ket_index[0]] if start_with_filling else []
                    inserted_ket_subsystem_qubits = [fillings_n_qubits[0]] if start_with_filling else []
                    fillings_ket_start_index = 1 if start_with_filling else 0

                    for i_of_index in range(len(origin_ket_index)):
                        inserted_ket_index.append(origin_ket_index[i_of_index])
                        inserted_ket_subsystem_qubits.append(subsystem_n_qubit_list[i_of_index])
                        if fillings_ket_start_index + i_of_index < len(fillings_ket_index):
                            inserted_ket_index.append(fillings_ket_index[fillings_ket_start_index + i_of_index])
                            inserted_ket_subsystem_qubits.append(
                                fillings_n_qubits[fillings_ket_start_index + i_of_index])

                    inserted_bra_index = [fillings_bra_index[0]] if start_with_filling else []
                    inserted_bra_subsystem_qubits = [fillings_n_qubits[0]] if start_with_filling else []
                    fillings_bra_start_index = 1 if start_with_filling else 0

                    for i_of_index in range(len(origin_bra_index)):
                        inserted_bra_index.append(origin_bra_index[i_of_index])
                        inserted_bra_subsystem_qubits.append(subsystem_n_qubit_list[i_of_index])
                        if fillings_bra_start_index + i_of_index < len(fillings_bra_index):
                            inserted_bra_index.append(fillings_bra_index[fillings_bra_start_index + i_of_index])
                            inserted_bra_subsystem_qubits.append(
                                fillings_n_qubits[fillings_bra_start_index + i_of_index])

                    eta_ket_basis = get_entry_index(inserted_ket_index, inserted_ket_subsystem_qubits)
                    eta_bra_basis = get_entry_index(inserted_bra_index, inserted_bra_subsystem_qubits)

                    eta_ket = get_ket(eta_ket_basis, output_qubits)
                    eta_bra = get_ket(eta_bra_basis, output_qubits).transpose().conj()

                    eta += coe * eta_ket @ eta_bra

    return eta


def get_prob_with_independent_intput_state(isometries: List[np.ndarray],
                                           input_states: List[np.ndarray],
                                           povm_operators: List[np.ndarray]) -> np.ndarray:
    """
    Get the output quantum state from an isometry represented process tensor with independent input state.
    :param isometries: the isometry represented process tensor. A list of isometries (np.ndarray).
    :param input_states: a list of input quantum states without mutual correlations.
    :param povm_operators: a list of POVM operators act on the output states.
    :return: quantum state output at the last time step.
    """

    n_time_steps = len(input_states)
    isometry_out_qubits = [round(np.log2(isometries[_t].shape[0])) for _t in range(n_time_steps)]
    output_qubits = [round(np.log2(povm_op.shape[0])) for povm_op in povm_operators]
    ancilla_qubits = [iso_out_q - out_q for (iso_out_q, out_q) in zip(isometry_out_qubits, output_qubits)]
    ancilla_state = 1
    tempo_output_with_ancilla = None
    for _t in range(n_time_steps):
        # Get current output state with ancilla

        tempo_output_with_ancilla = isometries[_t] @ np.kron(input_states[_t], ancilla_state) @ \
                                    isometries[_t].conj().transpose()
        if _t < n_time_steps - 1:
            n_out_qubits = output_qubits[_t]
            n_anc_qubits = ancilla_qubits[_t]
            op = np.kron(povm_operators[_t], np.identity(2 ** n_anc_qubits, dtype=complex))
            ancilla_state = la.partial_trace(op @ tempo_output_with_ancilla,
                                                       list(range(n_out_qubits)),
                                                       isometry_out_qubits[_t], 2)

    _t = n_time_steps - 1
    out_state = la.partial_trace(tempo_output_with_ancilla,
                                           list(range(output_qubits[_t], isometry_out_qubits[_t])),
                                           isometry_out_qubits[_t], 2)
    op = povm_operators[_t]
    prob = np.real(np.trace(op @ out_state))
    return prob


def get_anc_with_independent_intput_state(isometries: List[np.ndarray],
                                          input_states: List[np.ndarray],
                                          povm_operators: List[np.ndarray]) -> np.ndarray:
    """
    Get the output quantum state from an isometry represented process tensor with independent input state.
    :param isometries: the isometry represented process tensor. A list of isometries (np.ndarray).
    :param input_states: a list of input quantum states without mutual correlations.
    :param povm_operators: a list of POVM operators act on the output states.
    :return: quantum state output at the last time step.
    """

    n_time_steps = len(input_states)
    isometry_out_qubits = [round(np.log2(isometries[_t].shape[0])) for _t in range(n_time_steps)]
    output_qubits = [round(np.log2(povm_op.shape[0])) for povm_op in povm_operators]
    ancilla_qubits = [iso_out_q - out_q for (iso_out_q, out_q) in zip(isometry_out_qubits, output_qubits)]
    ancilla_state = 1
    tempo_output_with_ancilla = None
    for _t in range(n_time_steps):
        # Get current output state with ancilla

        tempo_output_with_ancilla = isometries[_t] @ np.kron(input_states[_t], ancilla_state) @ \
                                    isometries[_t].conj().transpose()
        if _t < n_time_steps - 1:
            n_out_qubits = output_qubits[_t]
            n_anc_qubits = ancilla_qubits[_t]
            op = np.kron(povm_operators[_t], np.identity(2 ** n_anc_qubits, dtype=complex))
            ancilla_state = la.partial_trace(op @ tempo_output_with_ancilla,
                                                       list(range(n_out_qubits)),
                                                       isometry_out_qubits[_t], 2)

    _t = n_time_steps - 1
    n_out_qubits = output_qubits[_t]
    n_anc_qubits = ancilla_qubits[_t]
    op = np.kron(povm_operators[_t], np.identity(2 ** n_anc_qubits, dtype=complex))
    anc = la.partial_trace(op @ tempo_output_with_ancilla,
                                     list(range(n_out_qubits)),
                                     isometry_out_qubits[_t], 2)

    return anc


def get_anc_with_instruments(isometries: List[np.ndarray], init_state: np.ndarray,
                             choi_states: List[np.ndarray]) -> np.ndarray:
    """
    Get the output quantum state from an isometry represented process tensor with independent input state.
    :param isometries: the isometry represented process tensor. A list of isometries (np.ndarray).
    :param input_states: a list of input quantum states without mutual correlations.
    :param povm_operators: a list of POVM operators act on the output states.
    :return: quantum state output at the last time step.
    """

    n_time_steps = len(choi_states)
    n_qubits = round(np.log2(init_state.shape[0]))
    isometry_out_qubits = [round(np.log2(isometries[_t].shape[0])) for _t in range(n_time_steps)]
    ancilla_qubits = [iso_out_q - n_qubits for iso_out_q in isometry_out_qubits]
    ancilla_state = init_state
    for _t in range(n_time_steps):
        # Get current output state with ancilla

        tempo_output_with_ancilla = isometries[_t] @ ancilla_state @ isometries[_t].conj().transpose()
        tempo_id_input_output_with_ancilla = np.kron(np.identity(2 ** n_qubits, dtype=complex),
                                                     tempo_output_with_ancilla)
        n_anc_qubits = ancilla_qubits[_t]
        op = np.kron(choi_states[_t], np.identity(2 ** n_anc_qubits, dtype=complex))
        ancilla_state = la.partial_trace(op @ tempo_id_input_output_with_ancilla,
                                                   [q + n_qubits for q in range(n_qubits)],
                                                   isometry_out_qubits[_t], 2)

    return ancilla_state


def get_out_state_with_instruments(isometries: List[np.ndarray], init_state: np.ndarray,
                                   choi_states: List[np.ndarray]) -> np.ndarray:
    """
    Get the output quantum state from an isometry represented process tensor with independent input state.
    :param isometries: the isometry represented process tensor. A list of isometries (np.ndarray).
    :param input_states: a list of input quantum states without mutual correlations.
    :param povm_operators: a list of POVM operators act on the output states.
    :return: quantum state output at the last time step.
    """

    n_time_steps = len(isometries)
    n_qubits = round(np.log2(init_state.shape[0]))
    isometry_out_qubits = [round(np.log2(isometries[_t].shape[0])) for _t in range(n_time_steps)]
    ancilla_qubits = [iso_out_q - n_qubits for iso_out_q in isometry_out_qubits]
    ancilla_state = init_state
    for _t in range(n_time_steps - 1):
        # Get current output state with ancilla

        tempo_output_with_ancilla = isometries[_t] @ ancilla_state @ isometries[_t].conj().transpose()
        tempo_id_input_output_with_ancilla = np.kron(np.identity(2 ** n_qubits, dtype=complex),
                                                     tempo_output_with_ancilla)
        n_anc_qubits = ancilla_qubits[_t]
        op = np.kron(choi_states[_t], np.identity(2 ** n_anc_qubits, dtype=complex))
        ancilla_state = la.partial_trace(op @ tempo_id_input_output_with_ancilla,
                                                   [q + n_qubits for q in range(n_qubits)],
                                                   isometry_out_qubits[_t] + n_qubits, 2)

    out_with_anc_state = isometries[-1] @ ancilla_state @ isometries[-1].conj().transpose()
    out_state = la.partial_trace(out_with_anc_state, [q + n_qubits for q in range(ancilla_qubits[-1])],
                                           isometry_out_qubits[-1], 2)

    return out_state


def get_full_output_state_with_independent_input_state(isometries: List[np.ndarray],
                                                       output_qubits: List[int],
                                                       input_states: List[np.ndarray]) -> np.ndarray:
    r"""
    Get full output quantum state from an isometry represented process tensor with arbitrary input state. No output
    dimension will be traced.

    .. math::

        V^{(k)}\in\mathcal{H}_{2k-2} \otimes \mathcal{H}_{A_{k-1}} \to \mathcal{H}_{2k-1} \otimes \mathcal{H}_{A_{k}}

    .. math::
        \rho_{out}\in\mathcal{H}_{1} \otimes \mathcal{H}_{3} \otimes \dots \otimes \mathcal{H}_{2N-1}

    .. math::

        V^{(k)}_{filled}\in \mathcal{H}_{1}\otimes \dots \otimes \mathcal{H}_{2k-3} \otimes \mathcal{H}_{2k-2}
        \otimes \mathcal{H}_{A_{k-1}} \to
        \mathcal{H}_{1}\otimes\dots\otimes\mathcal{H}_{2k-1} \otimes \mathcal{H}_{A_{k}}


    :param isometries: the isometry represented process tensor. A list of isometries (np.ndarray).
    :param output_qubits: a list of number which representing the number of qubits of output states.
    :param input_states: ndarray represented input state.
    :return: output state.
    """

    n_time_steps = len(isometries)

    filled_isometries = [copy.deepcopy(isometries[0])]
    for k in range(1, n_time_steps):
        V_filled = fill_isometry(isometry=isometries[k], in_qlist=[0], out_qlist=[0],
                                 filling_n_qubit_list=[sum(output_qubits[:k])])
        filled_isometries.append(V_filled)

    current_state = input_states[0]
    current_state = filled_isometries[0] @ current_state @ filled_isometries[0].transpose().conj()

    for k in range(1, n_time_steps):
        current_state = insert_operator(operator=current_state,
                                        qlist=[sum(output_qubits[:k])], fillings=[input_states[k]])

        current_state = filled_isometries[k] @ current_state @ filled_isometries[k].transpose().conj()

    n_qubits = round(np.log2(current_state.shape[0]))

    out_state = la.partial_trace(current_state,
                                           list(range(sum(output_qubits), n_qubits)),
                                           n_qubits, 2)
    return out_state


def get_full_output_state(isometries: List[np.ndarray], input_qubits: List[int],
                          output_qubits: List[int], input_state: np.ndarray) -> np.ndarray:
    r"""
    Get full output quantum state from an isometry represented process tensor with arbitrary input state. No output
    dimension will be traced.

    .. math::

        V^{(k)}\in\mathcal{H}_{2k-2} \otimes \mathcal{H}_{A_{k-1}} \to \mathcal{H}_{2k-1} \otimes \mathcal{H}_{A_{k}}

    .. math::

        \rho_{in}\in\mathcal{H}_{0} \otimes \mathcal{H}_{2} \otimes \dots \otimes \mathcal{H}_{2N-2}

    .. math::
        \rho_{out}\in\mathcal{H}_{1} \otimes \mathcal{H}_{3} \otimes \dots \otimes \mathcal{H}_{2N-1}

    .. math::

        V^{(k)}_{filled}\in \mathcal{H}_{1}\otimes \dots \otimes \mathcal{H}_{2k-3} \otimes \mathcal{H}_{2k-2}
        \otimes \mathcal{H}_{A_{k-1}} \to
        \mathcal{H}_{1}\otimes\dots\otimes\mathcal{H}_{2k-1} \otimes \mathcal{H}_{A_{k}}


    :param isometries: the isometry represented process tensor. A list of isometries (np.ndarray).
    :param input_qubits: a list of number which representing the number of qubits of intput states.
    :param output_qubits: a list of number which representing the number of qubits of output states.
    :param input_state: ndarray represented input state.
    :return: output state.
    """

    n_time_steps = len(isometries)

    filled_isometries = []

    for k in range(n_time_steps):
        V_filled = fill_isometry(isometry=isometries[k],
                                 in_qlist=[0, input_qubits[k]], out_qlist=[0, output_qubits[k]],
                                 filling_n_qubit_list=[sum(output_qubits[:k]), sum(input_qubits[k + 1:])])
        filled_isometries.append(V_filled)

    current_state = copy.deepcopy(input_state)
    for V_filled in filled_isometries:
        current_state = V_filled @ current_state @ V_filled.transpose().conj()

    n_qubits = round(np.log2(current_state.shape[0]))

    out_state = la.partial_trace(current_state,
                                           list(range(sum(output_qubits), n_qubits)),
                                           n_qubits, 2)

    return out_state


def isometries_to_choi_state(isometries: List[np.ndarray], input_qubits: List[int], output_qubits: List[int]):
    n_time_step = len(isometries)
    complete_in_qstate_list = []
    in_qstate_dual_set = []
    complete_povm_list = []
    povm_dual_set = []
    for _t in range(n_time_step):
        complete_pb = la.complete_pauli_matrices(input_qubits[_t])
        basis_size = len(complete_pb)
        dim = 2 ** input_qubits[_t]
        # construct complete pauli basis
        complete_pauli_matrix = [np.sqrt(2) ** (input_qubits[_t]) * pmat for pmat in complete_pb]

        complete_state_basis = [1 / dim * (complete_pauli_matrix[0] - complete_pauli_matrix[1])] + \
                               [1 / dim * (complete_pauli_matrix[0] + pmat) for pmat in complete_pauli_matrix[1:]]

        basis_mat = np.array([list(basis.ravel()) for basis in complete_state_basis], dtype=complex).transpose()
        dual_mat = np.linalg.pinv(basis_mat)
        dual_set = [dual_mat[i].reshape((dim, dim)) for i in range(basis_size)]

        in_qstate_dual_set.append(dual_set)
        complete_in_qstate_list.append(complete_state_basis)

    choi_state = np.zeros((2 ** sum(input_qubits + output_qubits), 2 ** sum(input_qubits + output_qubits)),
                          dtype=complex)
    for state_index in itertools.product(*tuple(list(range(4 ** input_qubits[_t])) for _t in range(n_time_step))):
        states = [complete_in_qstate_list[_t][state_index[_t]] for _t in range(n_time_step)]
        output_state = get_full_output_state_with_independent_input_state(isometries=isometries,
                                                                          output_qubits=output_qubits,
                                                                          input_states=states)
        input_dual_state = 1
        for _t in range(n_time_step):
            input_dual_state = np.kron(input_dual_state, in_qstate_dual_set[_t][state_index[_t]])
        choi_state += np.kron(output_state, input_dual_state)

    return choi_state


if __name__ == '__main__':
    # this block intend to test the correctness of functions in this file.
    test_input_qubits = [2, 2]
    test_output_qubits = [2, 2]

    # dim(H_{A_0}) = 0.
    test_n_ancilla = [0, 2, 4]

    test_isometries = [random_isometry(test_input_qubits[i] + test_n_ancilla[i],
                                       test_output_qubits[i] + test_n_ancilla[i + 1])
                       for i in range(len(test_input_qubits))]

    test_input_states = [random_state(q) for q in test_input_qubits]

    choi_state = isometries_to_choi_state(test_isometries, test_input_qubits, test_output_qubits)

    control_output = get_full_output_state_with_independent_input_state(test_isometries,
                                                                        test_output_qubits, test_input_states)

    state_tensor = 1
    for _state in test_input_states:
        state_tensor = np.kron(state_tensor, _state)

    identity_filling = np.identity(control_output.shape[0], dtype=complex)
    identity_filling = identity_filling
    choi_in = np.kron(identity_filling, state_tensor.transpose())
    full_output = la.partial_trace(choi_state @ choi_in, [3, 4, 5], 6, 2)
    control_output_2nd = get_full_output_state(test_isometries, test_input_qubits, test_output_qubits, state_tensor)

    print(np.linalg.norm(full_output - control_output))
    # print(np.trace(eta))

    check = 1
