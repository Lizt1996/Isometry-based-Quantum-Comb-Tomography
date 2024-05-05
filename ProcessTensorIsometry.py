import numpy as np
from typing import List, Tuple, Dict

from utils.pt_isometry import random_isometry, get_anc_with_independent_intput_state, get_anc_with_instruments

import itertools

import utils.la as la
import random
import copy
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from stiefel import get_stiefel_gradient_function, stiefel_optimization

class ProcTensorIsometryTomography:
    def __init__(self, input_states: List[List[np.ndarray]],
                 povm_operators: List[List[np.ndarray]], ancilla_qubits: List[int] = None):
        self._input_states = copy.deepcopy(input_states)
        self._povm_operators = copy.deepcopy(povm_operators)

        self._n_time_steps = len(input_states)

        self._input_qubits = [round(np.log2(_state[0].shape[0])) for _state in input_states]
        self._output_qubits = [round(np.log2(_povm[0].shape[0])) for _povm in povm_operators]

        self._state_basis_size = [len(state_basis) for state_basis in input_states]

        self._povm_basis_size = [len(povm_basis) for povm_basis in povm_operators]

        self._intput_dim = [2 ** _n for _n in self._input_qubits]
        self._output_dim = [2 ** _n for _n in self._output_qubits]

        self._ancilla_qubits = copy.deepcopy(ancilla_qubits)

        if self._ancilla_qubits is None:
            self._ancilla_qubits = [sum(self._input_qubits[:k + 1]) for k in range(self._n_time_steps)]

        self._isometry_input_qubits = [self._input_qubits[0]] + [self._input_qubits[i] + self._ancilla_qubits[i - 1] for
                                                                 i in range(1, self._n_time_steps)]
        self._isometry_output_qubits = [self._output_qubits[i] + self._ancilla_qubits[i]
                                        for i in range(self._n_time_steps)]

    @property
    def input_qlist(self):
        return self._input_qubits

    @property
    def output_qlist(self):
        return self._output_qubits

    def _get_eta(self, isometries: List[np.ndarray], input_index, povm_index):
        if len(isometries) == 0:
            return self._input_states[0][input_index[0]]
        input_states = [self._input_states[_t][input_index[_t]] for _t in range(len(isometries) + 1)]
        povm_ops = [self._povm_operators[_t][povm_index[_t]] for _t in range(len(isometries))]
        anc_state = get_anc_with_independent_intput_state(isometries, input_states[:-1], povm_ops)
        eta = np.kron(input_states[-1], anc_state)
        return eta

    def _get_sufficient_eta(self, target_isometry_input_qubits: int,
                            known_isometries: List[np.ndarray]):
        next_time_step = len(known_isometries)
        input_indexes = list(
            itertools.product(*tuple(list(range(self._state_basis_size[_i])) for _i in range(next_time_step + 1))))

        povm_indexes = list(
            itertools.product(*tuple(list(range(self._povm_basis_size[_i])) for _i in range(next_time_step))))

        index_ensemble = list(itertools.product(input_indexes, povm_indexes))

        etas = {}
        id_index = random.choice(list(range(len(index_ensemble))))
        eta = self._get_eta(isometries=known_isometries,
                            input_index=index_ensemble[id_index][0],
                            povm_index=index_ensemble[id_index][1])
        etas_mat = eta.reshape((-1, 1))
        etas[index_ensemble[id_index]] = eta
        index_ensemble.pop(id_index)

        rank = 1
        while rank < 4 ** target_isometry_input_qubits:
            id_index = random.choice(list(range(len(index_ensemble))))
            eta = self._get_eta(isometries=known_isometries,
                                input_index=index_ensemble[id_index][0],
                                povm_index=index_ensemble[id_index][1])
            tmp_eta_mat = np.concatenate([etas_mat, eta.reshape((-1, 1))], axis=-1)

            tmp_eta_rank = np.linalg.matrix_rank(tmp_eta_mat)
            if tmp_eta_rank > rank:
                rank = tmp_eta_rank
                etas_mat = tmp_eta_mat
                etas[index_ensemble[id_index]] = eta
            index_ensemble.pop(id_index)

        return etas

    def _get_all_etas(self, known_isometries: List[np.ndarray], last_full_eta: Dict):

        next_time_step = len(known_isometries)
        full_eta = {}
        if next_time_step == 0:
            for next_in in range(self._state_basis_size[0]):
                dual_index = (tuple([next_in]), tuple())
                full_eta[dual_index] = self._input_states[0][next_in]

            return full_eta

        n_anc_qubits = self._ancilla_qubits[next_time_step - 1]
        n_this_out_qubits = self._output_qubits[next_time_step - 1]

        this_povm_indexes = list(range(self._povm_basis_size[next_time_step - 1]))
        next_input_indexes = list(range(self._state_basis_size[next_time_step]))

        last_dual_indexes = last_full_eta.keys()

        for dual_index in last_dual_indexes:
            last_input_indexes = dual_index[0]
            last_output_indexes = dual_index[1]
            last_eta = last_full_eta[dual_index]
            tempo_output_with_ancilla = known_isometries[-1] @ last_eta @ known_isometries[-1].conj().transpose()

            in_out_index = list(itertools.product(next_input_indexes, this_povm_indexes))
            for next_in, this_out in in_out_index:
                op = np.kron(self._povm_operators[next_time_step - 1][this_out],
                             np.identity(2 ** n_anc_qubits,
                                         dtype=complex))

                ancilla_state = la.partial_trace(op @ tempo_output_with_ancilla,
                                                           list(range(n_this_out_qubits)),
                                                           n_anc_qubits + n_this_out_qubits, 2)

                eta = np.kron(self._input_states[next_time_step][next_in], ancilla_state)

                input_indexes = tuple(list(last_input_indexes) + [next_in])
                output_indexes = tuple(list(last_output_indexes) + [this_out])

                full_eta[(input_indexes, output_indexes)] = eta

        return full_eta

    def fit_euclidian_constrained(self, measurement_data: List[Dict[Tuple[Tuple[int], Tuple[int]], float]], **kwargs):
        max_iteration = kwargs.get('max_iteration', 10000)
        eps = kwargs.get('eps', 1e-12)
        ftol = kwargs.get('ftol', 1e-12)

        mea_data = copy.deepcopy(measurement_data)
        reconstructed_isometries = []
        for _t in range(self._n_time_steps):
            utilizing_data = mea_data[_t]
            isometry_size = (self._isometry_output_qubits[_t], self._isometry_input_qubits[_t])
            etas_size = tuple([self._state_basis_size[:_t + 1] + self._povm_basis_size[:_t] + list(isometry_size)])

            etas = self._get_sufficient_eta(self._isometry_input_qubits[_t], reconstructed_isometries[:_t])

            _input_dim = 2 ** self._isometry_input_qubits[_t]
            _output_dim = 2 ** self._isometry_output_qubits[_t]

            n_pars = 2 * _input_dim * _output_dim

            def _cost_fn(params):
                _V_real = params[:_input_dim * _output_dim].reshape((_output_dim, _input_dim))
                _V_image = params[_input_dim * _output_dim:].reshape((_output_dim, _input_dim))
                _V = _V_real + 1j * _V_image
                _cost = 0
                for tip in etas.keys():
                    eta = etas[tip]
                    eta_t = _V @ eta @ _V.conj().transpose()
                    for last_povm_index in range(len(self._povm_operators[_t])):
                        op = np.kron(self._povm_operators[_t][last_povm_index],
                                     np.identity(2 ** self._ancilla_qubits[_t], dtype=complex))
                        input_index = tip[0]
                        povm_index = tuple(list(tip[1]) + [last_povm_index])
                        measured_prob = mea_data[_t][(input_index, povm_index)]
                        recovered_prob = np.trace(op @ eta_t)
                        _cost += np.real(recovered_prob - measured_prob) ** 2
                return _cost

            def _constraints(params):
                _V_real = params[:_input_dim * _output_dim].reshape((_output_dim, _input_dim))
                _V_image = params[_input_dim * _output_dim:].reshape((_output_dim, _input_dim))
                _V = _V_real + 1j * _V_image

                zero_mat = _V.conj().transpose() @ _V - np.identity(2 ** self._isometry_input_qubits[_t], dtype=complex)

                return [np.linalg.norm(zero_mat)]

                # eq_list = []
                # for i in range(zero_mat.shape[0]):
                #     for j in range(zero_mat.shape[1]):
                #         eq_list.append(np.real(zero_mat[i][j]))
                # return eq_list

            initial_V = random_isometry(self._isometry_input_qubits[_t], self._isometry_output_qubits[_t])
            x0 = np.array(list(np.real(initial_V.ravel())) + list(np.imag(initial_V.ravel())), dtype=float)

            def _cb_fn(x):
                print(_cost_fn(x))

            res = minimize(fun=_cost_fn,
                           x0=x0,
                           constraints={'type': 'eq', 'fun': _constraints},
                           method='SLSQP',
                           options={'disp': True, 'maxiter': max_iteration, 'eps': eps, 'ftol': ftol},
                           callback=_cb_fn
                           )

            res_x = res.x
            V_real = res_x[:_input_dim * _output_dim].reshape((_output_dim, _input_dim))
            V_image = res_x[_input_dim * _output_dim:].reshape((_output_dim, _input_dim))
            V = V_real + 1j * V_image

            reconstructed_isometries.append(V)

        return reconstructed_isometries

    def fit_euclidian_penalty(self, measurement_data: List[Dict[Tuple[Tuple[int], Tuple[int]], float]], **kwargs):
        penalty_coe = kwargs.get('penalty_coe', 1)

        from stiefel import euclidian_optimization

        mea_data = copy.deepcopy(measurement_data)
        reconstructed_isometries = []
        for _t in range(self._n_time_steps):
            utilizing_data = mea_data[_t]
            isometry_size = (self._isometry_output_qubits[_t], self._isometry_input_qubits[_t])
            etas_size = tuple([self._state_basis_size[:_t + 1] + self._povm_basis_size[:_t] + list(isometry_size)])

            etas = self._get_sufficient_eta(self._isometry_input_qubits[_t], reconstructed_isometries[:_t])

            _input_dim = 2 ** self._isometry_input_qubits[_t]
            _output_dim = 2 ** self._isometry_output_qubits[_t]

            n_pars = 2 * _input_dim * _output_dim

            def _cost_fn(_V):
                _cost = 0
                for tip in etas.keys():
                    eta = etas[tip]
                    eta_t = _V @ eta @ _V.conj().transpose()
                    for last_povm_index in range(len(self._povm_operators[_t])):
                        op = np.kron(self._povm_operators[_t][last_povm_index],
                                     np.identity(2 ** self._ancilla_qubits[_t], dtype=complex))
                        input_index = tip[0]
                        povm_index = tuple(list(tip[1]) + [last_povm_index])
                        measured_prob = mea_data[_t][(input_index, povm_index)]
                        recovered_prob = np.trace(op @ eta_t)
                        _cost += np.real(recovered_prob - measured_prob) ** 2
                return _cost

            def _cost_fn_with_penalty(_V):

                _cost = 0
                for tip in etas.keys():
                    eta = etas[tip]
                    eta_t = _V @ eta @ _V.conj().transpose()
                    for last_povm_index in range(len(self._povm_operators[_t])):
                        op = np.kron(self._povm_operators[_t][last_povm_index],
                                     np.identity(2 ** self._ancilla_qubits[_t], dtype=complex))
                        input_index = tip[0]
                        povm_index = tuple(list(tip[1]) + [last_povm_index])
                        measured_prob = mea_data[_t][(input_index, povm_index)]
                        recovered_prob = np.trace(op @ eta_t)
                        _cost += np.real(recovered_prob - measured_prob) ** 2

                zero_mat = _V.conj().transpose() @ _V - np.identity(2 ** self._isometry_input_qubits[_t], dtype=complex)

                return _cost + np.real(penalty_coe * np.trace(zero_mat.conj().transpose() @ zero_mat))

            def _euclidian_gradient(_V: np.ndarray):
                _G = np.zeros(_V.shape, dtype=complex)
                for tip in etas.keys():
                    eta = etas[tip]
                    eta_t = _V @ eta @ _V.conj().transpose()
                    _V_eta = _V @ eta

                    for last_povm_index in range(len(self._povm_operators[_t])):
                        # get measurement probability
                        input_index = tip[0]
                        povm_index = tuple(list(tip[1]) + [last_povm_index])
                        measured_prob = mea_data[_t][(input_index, povm_index)]

                        # compute recovered probability
                        op = np.kron(self._povm_operators[_t][last_povm_index],
                                     np.identity(2 ** self._ancilla_qubits[_t], dtype=complex))
                        recovered_prob = np.trace(op @ eta_t)

                        # compute gradient w.r.t. the index
                        # -2 * (measured_prob - recovered_prob) * op @ _V @ eta
                        _g = -2 * (measured_prob - recovered_prob) * op @ _V_eta
                        _G += _g

                # penalty derivative
                _G += penalty_coe * (4 * _V @ _V.conj().transpose() @ _V - 2 * _V)
                return _G

            W0 = random_isometry(self._isometry_input_qubits[_t], self._isometry_output_qubits[_t])

            def _cb_fn(x):
                print(_cost_fn(x))

            W = euclidian_optimization(cost_fun=_cost_fn_with_penalty, euclidian_grad_fn=_euclidian_gradient,
                                       X0=W0, callback=_cb_fn, **kwargs)

            reconstructed_isometries.append(W)
        gcost = self.gross_cost(measurement_data, reconstructed_isometries)
        return reconstructed_isometries, gcost

    def fit_stiefel_SGD(self, measurement_data: List[Dict[Tuple[Tuple[int], Tuple[int]], float]], **kwargs):
        alpha = kwargs.get('alpha', 0.01)
        tol = kwargs.get('tol', 1e-6)
        max_iterations = kwargs.get('max_iterations', 10000)
        method = kwargs.get('method', 'gradient_decent')

        mea_data = copy.deepcopy(measurement_data)
        reconstructed_isometries = []
        for _t in range(self._n_time_steps):
            utilizing_data = mea_data[_t]
            isometry_size = (self._isometry_output_qubits[_t], self._isometry_input_qubits[_t])
            etas_size = tuple([self._state_basis_size[:_t + 1] + self._povm_basis_size[:_t] + list(isometry_size)])

            etas = self._get_sufficient_eta(self._isometry_input_qubits[_t], reconstructed_isometries[:_t])

            _input_dim = 2 ** self._isometry_input_qubits[_t]
            _output_dim = 2 ** self._isometry_output_qubits[_t]

            n_pars = 2 * _input_dim * _output_dim

            def _cost_fn(_V: np.ndarray):
                _cost = 0
                for tip in etas.keys():
                    eta = etas[tip]
                    eta_t = _V @ eta @ _V.conj().transpose()
                    for last_povm_index in range(len(self._povm_operators[_t])):
                        op = np.kron(self._povm_operators[_t][last_povm_index],
                                     np.identity(2 ** self._ancilla_qubits[_t], dtype=complex))
                        input_index = tip[0]
                        povm_index = tuple(list(tip[1]) + [last_povm_index])
                        measured_prob = mea_data[_t][(input_index, povm_index)]
                        recovered_prob = np.trace(op @ eta_t)
                        _cost += np.real(recovered_prob - measured_prob) ** 2
                return _cost

            def _euclidian_gradient(_V: np.ndarray):
                _G = np.zeros(_V.shape, dtype=complex)
                for tip in etas.keys():
                    eta = etas[tip]
                    eta_t = _V @ eta @ _V.conj().transpose()
                    _V_eta = _V @ eta

                    for last_povm_index in range(len(self._povm_operators[_t])):
                        # get measurement probability
                        input_index = tip[0]
                        povm_index = tuple(list(tip[1]) + [last_povm_index])
                        measured_prob = mea_data[_t][(input_index, povm_index)]

                        # compute recovered probability
                        op = np.kron(self._povm_operators[_t][last_povm_index],
                                     np.identity(2 ** self._ancilla_qubits[_t], dtype=complex))
                        recovered_prob = np.trace(op @ eta_t)

                        # compute gradient w.r.t. the index
                        # -2 * (measured_prob - recovered_prob) * op @ _V @ eta
                        _g = -2 * (measured_prob - recovered_prob) * op @ _V_eta
                        _G += _g
                return _G

            W0 = random_isometry(self._isometry_input_qubits[_t], self._isometry_output_qubits[_t])

            stiefel_grad_fn = get_stiefel_gradient_function(_euclidian_gradient)
            W = stiefel_optimization(cost_fun=_cost_fn, stiefel_grad_fn=stiefel_grad_fn,
                                     X0=W0, **kwargs)

            print('time_step = ' + str(_t) + ' cost = ' + str(_cost_fn(W)))

            reconstructed_isometries.append(W)

        return reconstructed_isometries

    def fit_stiefel_least(self, measurement_data: List[Dict[Tuple[Tuple[int], Tuple[int]], float]], **kwargs):
        alpha = kwargs.get('alpha', 0.01)
        tol = kwargs.get('tol', 1e-6)
        max_iterations = kwargs.get('max_iterations', 10000)
        method = kwargs.get('method', 'gradient_decent')

        mea_data = copy.deepcopy(measurement_data)
        reconstructed_isometries = []
        for _t in range(self._n_time_steps):
            utilizing_data = mea_data[_t]
            isometry_size = (self._isometry_output_qubits[_t], self._isometry_input_qubits[_t])
            etas_size = tuple([self._state_basis_size[:_t + 1] + self._povm_basis_size[:_t] + list(isometry_size)])

            etas = self._get_sufficient_eta(self._isometry_input_qubits[_t], reconstructed_isometries[:_t])

            _input_dim = 2 ** self._isometry_input_qubits[_t]
            _output_dim = 2 ** self._isometry_output_qubits[_t]

            n_pars = 2 * _input_dim * _output_dim

            def _cost_fn(_V: np.ndarray):
                _cost = 0
                for tip in etas.keys():
                    eta = etas[tip]
                    eta_t = _V @ eta @ _V.conj().transpose()
                    for last_povm_index in range(len(self._povm_operators[_t])):
                        op = np.kron(self._povm_operators[_t][last_povm_index],
                                     np.identity(2 ** self._ancilla_qubits[_t], dtype=complex))
                        input_index = tip[0]
                        povm_index = tuple(list(tip[1]) + [last_povm_index])
                        measured_prob = mea_data[_t][(input_index, povm_index)]
                        recovered_prob = np.trace(op @ eta_t)
                        _cost += np.real(recovered_prob - measured_prob) ** 2
                return _cost

            def _euclidian_gradient(_V: np.ndarray):
                _G = np.zeros(_V.shape, dtype=complex)
                for tip in etas.keys():
                    eta = etas[tip]
                    eta_t = _V @ eta @ _V.conj().transpose()
                    _V_eta = _V @ eta

                    for last_povm_index in range(len(self._povm_operators[_t])):
                        # get measurement probability
                        input_index = tip[0]
                        povm_index = tuple(list(tip[1]) + [last_povm_index])
                        measured_prob = mea_data[_t][(input_index, povm_index)]

                        # compute recovered probability
                        op = np.kron(self._povm_operators[_t][last_povm_index],
                                     np.identity(2 ** self._ancilla_qubits[_t], dtype=complex))
                        recovered_prob = np.trace(op @ eta_t)

                        # compute gradient w.r.t. the index
                        # -2 * (measured_prob - recovered_prob) * op @ _V @ eta
                        _g = -2 * (measured_prob - recovered_prob) * op @ _V_eta
                        _G += _g
                return _G

            W0 = random_isometry(self._isometry_input_qubits[_t], self._isometry_output_qubits[_t])

            stiefel_grad_fn = get_stiefel_gradient_function(_euclidian_gradient)
            W = stiefel_optimization(cost_fun=_cost_fn, euclidian_grad_fn=_euclidian_gradient,
                                     stiefel_grad_fn=stiefel_grad_fn,
                                     X0=W0, **kwargs)

            print('time_step = ' + str(_t) + ' cost = ' + str(_cost_fn(W)))

            reconstructed_isometries.append(W)

        gcost = self.gross_cost(measurement_data, reconstructed_isometries)

        return reconstructed_isometries, gcost

    def fit_stiefel(self, measurement_data: List[Dict[Tuple[Tuple[int], Tuple[int]], float]], **kwargs):
        alpha = kwargs.get('alpha', 0.01)
        tol = kwargs.get('tol', 1e-6)
        max_iterations = kwargs.get('max_iterations', 10000)
        method = kwargs.get('method', 'gradient_decent')

        mea_data = copy.deepcopy(measurement_data)
        reconstructed_isometries = []
        etas = {}

        for _t in range(self._n_time_steps):
            utilizing_data = mea_data[_t]
            isometry_size = (self._isometry_output_qubits[_t], self._isometry_input_qubits[_t])
            etas_size = tuple([self._state_basis_size[:_t + 1] + self._povm_basis_size[:_t] + list(isometry_size)])

            etas = self._get_all_etas(reconstructed_isometries[:_t], etas)

            _input_dim = 2 ** self._isometry_input_qubits[_t]
            _output_dim = 2 ** self._isometry_output_qubits[_t]

            n_pars = 2 * _input_dim * _output_dim

            def _cost_fn(_V: np.ndarray):
                _cost = 0
                for tip in etas.keys():
                    eta = etas[tip]
                    eta_t = _V @ eta @ _V.conj().transpose()
                    for last_povm_index in range(len(self._povm_operators[_t])):
                        op = np.kron(self._povm_operators[_t][last_povm_index],
                                     np.identity(2 ** self._ancilla_qubits[_t], dtype=complex))
                        input_index = tip[0]
                        povm_index = tuple(list(tip[1]) + [last_povm_index])
                        measured_prob = mea_data[_t][(input_index, povm_index)]
                        recovered_prob = np.trace(op @ eta_t)
                        _cost += np.real(recovered_prob - measured_prob) ** 2
                return _cost

            def _euclidian_gradient(_V: np.ndarray):
                _G = np.zeros(_V.shape, dtype=complex)
                for tip in etas.keys():
                    eta = etas[tip]
                    eta_t = _V @ eta @ _V.conj().transpose()
                    _V_eta = _V @ eta

                    for last_povm_index in range(len(self._povm_operators[_t])):
                        # get measurement probability
                        input_index = tip[0]
                        povm_index = tuple(list(tip[1]) + [last_povm_index])
                        measured_prob = mea_data[_t][(input_index, povm_index)]

                        # compute recovered probability
                        op = np.kron(self._povm_operators[_t][last_povm_index],
                                     np.identity(2 ** self._ancilla_qubits[_t], dtype=complex))
                        recovered_prob = np.trace(op @ eta_t)

                        # compute gradient w.r.t. the index
                        # -2 * (measured_prob - recovered_prob) * op @ _V @ eta
                        _g = -2 * (measured_prob - recovered_prob) * op @ _V_eta
                        _G += _g
                return _G

            W0 = random_isometry(self._isometry_input_qubits[_t], self._isometry_output_qubits[_t])

            stiefel_grad_fn = get_stiefel_gradient_function(_euclidian_gradient)
            W = stiefel_optimization(cost_fun=_cost_fn, euclidian_grad_fn=_euclidian_gradient,
                                     stiefel_grad_fn=stiefel_grad_fn,
                                     X0=W0, **kwargs)

            print('time_step = ' + str(_t) + ' cost = ' + str(_cost_fn(W)))

            reconstructed_isometries.append(W)

        gcost = self.gross_cost(measurement_data, reconstructed_isometries)

        return reconstructed_isometries, gcost

    def fit_lagrange(self, measurement_data: List[Dict[Tuple[Tuple[int], Tuple[int]], float]], **kwargs):
        max_iteration = kwargs.get('max_iteration', 10000)
        eps = kwargs.get('eps', 1e-6)
        tol = kwargs.get('tol', 1e-4)
        penalty_coe = kwargs.get('penalty_coe', 100)

        mea_data = copy.deepcopy(measurement_data)
        reconstructed_isometries = []
        for _t in range(self._n_time_steps):
            utilizing_data = mea_data[_t]
            isometry_size = (self._isometry_output_qubits[_t], self._isometry_input_qubits[_t])
            etas_size = tuple([self._state_basis_size[:_t + 1] + self._povm_basis_size[:_t] + list(isometry_size)])

            etas = self._get_sufficient_eta(self._isometry_input_qubits[_t], reconstructed_isometries[:_t])

            _input_dim = 2 ** self._isometry_input_qubits[_t]
            _output_dim = 2 ** self._isometry_output_qubits[_t]

            n_pars = 2 * _input_dim * _output_dim

            def _cost_fn_la(params):
                _V_real = params[:_input_dim * _output_dim].reshape((_output_dim, _input_dim))
                _V_image = params[_input_dim * _output_dim:-1].reshape((_output_dim, _input_dim))
                _V = _V_real + 1j * _V_image
                con_lambda = params[-1]

                _cost = 0
                for tip in etas.keys():
                    eta = etas[tip]
                    eta_t = _V @ eta @ _V.conj().transpose()
                    for last_povm_index in range(len(self._povm_operators[_t])):
                        op = np.kron(self._povm_operators[_t][last_povm_index],
                                     np.identity(2 ** self._ancilla_qubits[_t], dtype=complex))
                        input_index = tip[0]
                        povm_index = tuple(list(tip[1]) + [last_povm_index])
                        measured_prob = mea_data[_t][(input_index, povm_index)]
                        recovered_prob = np.trace(op @ eta_t)
                        _cost += np.real(recovered_prob - measured_prob) ** 2

                    constraint = np.linalg.norm(
                        _V.conj().transpose() @ _V - np.identity(2 ** self._isometry_input_qubits[_t], dtype=complex))
                    _cost += con_lambda * constraint

                return _cost

            def _lambda_constraint(param):
                return [param[-1] - 1e-3]

            # initial_V = np.zeros((_output_dim, _input_dim), dtype=float)
            # for i in range(min(_output_dim,_input_dim)):
            #     initial_V[i][i] = 1
            # x0 = np.array(list(initial_V.ravel()) + [0 for _ in range(_input_dim*_output_dim)], dtype=float)

            # x0 = np.random.uniform(-1, 1, n_pars)

            initial_V = random_isometry(self._isometry_input_qubits[_t], self._isometry_output_qubits[_t])
            lamb0 = 10
            ini_x0 = list(np.real(initial_V.ravel())) + list(np.imag(initial_V.ravel()))
            ini_x0 = ini_x0 + [lamb0]
            x0 = np.array(ini_x0, dtype=float)

            def _cb_fn(x):
                print('Lagrange: cost = ' + str(_cost_fn_la(x)))
                print(x[-1])

            res = minimize(fun=_cost_fn_la,
                           x0=x0,
                           constraints={'type': 'ineq', 'fun': _lambda_constraint},
                           options={'disp': True, 'maxiter': max_iteration},
                           callback=_cb_fn
                           )

            res_x = res.x
            V_real = res_x[:_input_dim * _output_dim].reshape((_output_dim, _input_dim))
            V_image = res_x[_input_dim * _output_dim:-1].reshape((_output_dim, _input_dim))
            V = V_real + 1j * V_image

            reconstructed_isometries.append(V)

        return reconstructed_isometries

    def test_ideal_isometry(self, measurement_data: List[Dict[Tuple[Tuple[int], Tuple[int]], float]], ideal_isometries):

        mea_data = copy.deepcopy(measurement_data)
        reconstructed_isometries = []
        for _t in range(self._n_time_steps):
            utilizing_data = mea_data[_t]
            isometry_size = (self._isometry_output_qubits[_t], self._isometry_input_qubits[_t])
            etas_size = tuple([self._state_basis_size[:_t + 1] + self._povm_basis_size[:_t] + list(isometry_size)])

            etas = self._get_sufficient_eta(self._isometry_input_qubits[_t], reconstructed_isometries[:_t])

            _input_dim = 2 ** self._isometry_input_qubits[_t]
            _output_dim = 2 ** self._isometry_output_qubits[_t]

            n_pars = 2 * _input_dim * _output_dim

            def _cost_fn(params):
                _V_real = params[:_input_dim * _output_dim].reshape((_output_dim, _input_dim))
                _V_image = params[_input_dim * _output_dim:].reshape((_output_dim, _input_dim))
                _V = _V_real + 1j * _V_image
                _cost = 0
                for tip in etas.keys():
                    eta = etas[tip]
                    eta_t = _V @ eta @ _V.conj().transpose()
                    for last_povm_index in range(len(self._povm_operators[_t])):
                        op = np.kron(self._povm_operators[_t][last_povm_index],
                                     np.identity(2 ** self._ancilla_qubits[_t], dtype=complex))
                        input_index = tip[0]
                        povm_index = tuple(list(tip[1]) + [last_povm_index])
                        measured_prob = mea_data[_t][(input_index, povm_index)]
                        recovered_prob = np.trace(op @ eta_t)
                        _cost += np.real(recovered_prob - measured_prob) ** 2
                return _cost

            def _constraints(params):
                _V_real = params[:_input_dim * _output_dim].reshape((_output_dim, _input_dim))
                _V_image = params[_input_dim * _output_dim:].reshape((_output_dim, _input_dim))
                _V = _V_real + _V_image

                zero_mat = _V.conj().transpose() @ _V - np.identity(2 ** self._isometry_input_qubits[_t], dtype=complex)

                eq_list = []
                for i in range(zero_mat.shape[0]):
                    for j in range(zero_mat.shape[1]):
                        eq_list.append(np.real(zero_mat[i][j]))
                return eq_list

            V = ideal_isometries[_t]

            V_real = np.real(V)
            V_image = np.imag(V)

            x0 = np.array(list(V_real.ravel()) + list(V_image.ravel()), dtype=float)

            print(_cost_fn(x0))
            reconstructed_isometries.append(V)

    def gross_cost(self, measurement_data, isometries):
        mea_data = copy.deepcopy(measurement_data)
        gross_cost_val = 0
        for _t in range(self._n_time_steps):

            etas = self._get_sufficient_eta(self._isometry_input_qubits[_t], isometries[:_t])

            _input_dim = 2 ** self._isometry_input_qubits[_t]
            _output_dim = 2 ** self._isometry_output_qubits[_t]

            def _cost_fn(_V: np.ndarray):
                _cost = 0
                for tip in etas.keys():
                    eta = etas[tip]
                    eta_t = _V @ eta @ _V.conj().transpose()
                    for last_povm_index in range(len(self._povm_operators[_t])):
                        op = np.kron(self._povm_operators[_t][last_povm_index],
                                     np.identity(2 ** self._ancilla_qubits[_t], dtype=complex))
                        input_index = tip[0]
                        povm_index = tuple(list(tip[1]) + [last_povm_index])
                        measured_prob = mea_data[_t][(input_index, povm_index)]
                        recovered_prob = np.trace(op @ eta_t)
                        _cost += np.real(recovered_prob - measured_prob) ** 2
                return _cost

            gross_cost_val += _cost_fn(isometries[_t])

        return gross_cost_val


class InstProcTensorIsoTomography:
    def __init__(self, initial_states: List[np.ndarray], cptp_operations: List[List[np.ndarray]],
                 povm_operators: List[List[np.ndarray]], ancilla_qubits: List[int] = None):
        self._initial_states = copy.deepcopy(initial_states)
        self._povm_operators = copy.deepcopy(povm_operators)

        # CPTP operations A(\rho) that represented by Choi state \xi such that Tr_{in} \xi (I\otimes \rho) = A(\rho)
        self._cptp_operations = copy.deepcopy(cptp_operations)
        self._n_time_steps = len(cptp_operations) + 1

        self._n_qubits = round(np.log2(self._initial_states[0].shape[0]))
        self._input_qubits = [self._n_qubits for _ in range(self._n_time_steps)]
        self._output_qubits = [self._n_qubits for _ in range(self._n_time_steps)]
        self._dim = 2 ** self._n_qubits

        # required CPTP operations and POVM operators
        self._n_cptp_ops = (self._dim ** 2) * (self._dim ** 2 - 1)
        self._n_povm_ops = self._dim ** 2

        # the initial states that the device can provide.
        # We do not require the initial state to be complete
        # because most NISQ devices can only provide |0><0|
        self._n_init_states = len(initial_states)

        self._ancilla_qubits = copy.deepcopy(ancilla_qubits)

        if self._ancilla_qubits is None:
            self._ancilla_qubits = [self._n_qubits * (k + 1) for k in range(self._n_time_steps)]

        self._isometry_input_qubits = [self._n_qubits]

        [self._isometry_input_qubits.append(self._n_qubits + self._ancilla_qubits[i]) for i in range(self._n_time_steps-1)]
        self._isometry_output_qubits = [self._n_qubits + self._ancilla_qubits[k] for k in range(self._n_time_steps)]

    @property
    def input_qlist(self):
        return self._input_qubits

    @property
    def output_qlist(self):
        return self._output_qubits

    def _get_eta(self, isometries: List[np.ndarray], init_state_index, cptp_index):
        if len(isometries) == 0:
            return self._initial_states[init_state_index]
        init_state = self._initial_states[init_state_index]
        cptp_choi_states = [self._cptp_operations[_t][cptp_index[_t]] for _t in range(len(isometries))]
        eta = get_anc_with_instruments(isometries, init_state=init_state, choi_states=cptp_choi_states)
        return eta

    def _get_sufficient_eta(self, target_isometry_input_qubits: int,
                            known_isometries: List[np.ndarray]):
        next_time_step = len(known_isometries)
        init_state_indexes = list(range(self._n_init_states))
        cptp_indexes = list(
            itertools.product(*tuple(list(range(self._n_cptp_ops)) for _i in range(next_time_step))))

        index_ensemble = list(itertools.product(init_state_indexes, cptp_indexes))

        etas = {}
        id_index = random.choice(list(range(len(index_ensemble))))
        eta = self._get_eta(isometries=known_isometries,
                            init_state_index=index_ensemble[id_index][0],
                            cptp_index=index_ensemble[id_index][1])
        etas_mat = eta.reshape((-1, 1))
        etas[index_ensemble[id_index]] = eta
        index_ensemble.pop(id_index)

        rank = 1
        while rank < 4 ** target_isometry_input_qubits:
            id_index = random.choice(list(range(len(index_ensemble))))
            eta = self._get_eta(isometries=known_isometries,
                                init_state_index=index_ensemble[id_index][0],
                                cptp_index=index_ensemble[id_index][1])
            tmp_eta_mat = np.concatenate([etas_mat, eta.reshape((-1, 1))], axis=-1)

            tmp_eta_rank = np.linalg.matrix_rank(tmp_eta_mat)
            if tmp_eta_rank > rank:
                rank = tmp_eta_rank
                etas_mat = tmp_eta_mat
                etas[index_ensemble[id_index]] = eta
            index_ensemble.pop(id_index)

        return etas

    def _get_all_etas(self, known_isometries: List[np.ndarray], last_full_eta: Dict):

        next_time_step = len(known_isometries)
        full_eta = {}
        if next_time_step == 0:
            for init_state_i in range(self._n_init_states):
                dual_index = (init_state_i, tuple())
                full_eta[dual_index] = self._initial_states[init_state_i]

            return full_eta

        n_anc_qubits = self._ancilla_qubits[next_time_step - 1]
        n_this_out_qubits = self._output_qubits[next_time_step - 1]

        next_inst_indexes = list(range(len(self._cptp_operations[next_time_step - 1])))

        last_dual_indexes = last_full_eta.keys()

        for dual_index in last_dual_indexes:
            init_state_index = dual_index[0]
            last_cptp_index = dual_index[1]
            last_eta = last_full_eta[dual_index]
            tempo_output_with_ancilla = known_isometries[-1] @ last_eta @ known_isometries[-1].conj().transpose()

            for next_inst_index in next_inst_indexes:
                choi_state = self._cptp_operations[next_time_step - 1][next_inst_index]

                tempo_id_input_output_with_ancilla = np.kron(np.identity(2 ** self._n_qubits, dtype=complex),
                                                             tempo_output_with_ancilla)

                op = np.kron(choi_state, np.identity(2 ** n_anc_qubits, dtype=complex))
                eta = la.partial_trace(op @ tempo_id_input_output_with_ancilla,
                                                 [q + self._n_qubits for q in range(self._n_qubits)],
                                                 n_anc_qubits + n_this_out_qubits + self._n_qubits, 2)

                cptp_index = tuple(list(last_cptp_index) + [next_inst_index])

                full_eta[(init_state_index, cptp_index)] = eta

        return full_eta

    def fit_stiefel(self, measurement_data: List[Dict[Tuple[int, Tuple[int], int], float]], **kwargs):
        alpha = kwargs.get('alpha', 0.01)
        tol = kwargs.get('tol', 1e-6)
        max_iterations = kwargs.get('max_iterations', 10000)
        method = kwargs.get('method', 'gradient_decent')

        mea_data = copy.deepcopy(measurement_data)
        reconstructed_isometries = []
        etas = {}
        for _t in range(self._n_time_steps):
            etas = self._get_all_etas(reconstructed_isometries[:_t], etas)
            _input_dim = 2 ** self._isometry_input_qubits[_t]
            _output_dim = 2 ** self._isometry_output_qubits[_t]

            def _cost_fn(_V: np.ndarray):
                _cost = 0
                for tip in etas.keys():
                    eta = etas[tip]
                    eta_t = _V @ eta @ _V.conj().transpose()
                    for povm_index in range(len(self._povm_operators[_t])):
                        op = np.kron(self._povm_operators[_t][povm_index],
                                     np.identity(2 ** self._ancilla_qubits[_t], dtype=complex))
                        init_state_index = tip[0]
                        cptp_index = tip[1]

                        measured_prob = mea_data[_t][(init_state_index, cptp_index, povm_index)]
                        recovered_prob = np.real(np.trace(op @ eta_t))
                        _cost += np.real(recovered_prob - measured_prob) ** 2
                return _cost

            def _euclidian_gradient(_V: np.ndarray):
                _G = np.zeros(_V.shape, dtype=complex)
                for tip in etas.keys():
                    eta = etas[tip]
                    eta_t = _V @ eta @ _V.conj().transpose()
                    _V_eta = _V @ eta

                    for povm_index in range(len(self._povm_operators[_t])):
                        # get measurement probability
                        init_state_index = tip[0]
                        cptp_index = tip[1]

                        measured_prob = mea_data[_t][(init_state_index, cptp_index, povm_index)]

                        # compute recovered probability
                        op = np.kron(self._povm_operators[_t][povm_index],
                                     np.identity(2 ** self._ancilla_qubits[_t], dtype=complex))
                        recovered_prob = np.real(np.trace(op @ eta_t))

                        # compute gradient w.r.t. the index
                        # -2 * (measured_prob - recovered_prob) * op @ _V @ eta
                        _g = -2 * (measured_prob - recovered_prob) * op @ _V_eta
                        _G += _g
                return _G

            W0 = np.zeros((_output_dim, _input_dim), dtype=complex)
            for i in range(_input_dim):
                W0[i][i] = 1. + 0.j
            # W0 = random_isometry(self._isometry_input_qubits[_t], self._isometry_output_qubits[_t])

            stiefel_grad_fn = get_stiefel_gradient_function(_euclidian_gradient)
            W = stiefel_optimization(cost_fun=_cost_fn, euclidian_grad_fn=_euclidian_gradient,
                                     stiefel_grad_fn=stiefel_grad_fn,
                                     X0=W0, **kwargs)

            print('time_step = ' + str(_t) + ' cost = ' + str(_cost_fn(W)))

            reconstructed_isometries.append(W)

        return reconstructed_isometries

    def test_ideal_isometry(self, measurement_data, ideal_isometries):

        mea_data = copy.deepcopy(measurement_data)
        etas = {}
        cost_vals = []
        for _t in range(self._n_time_steps):
            etas = self._get_all_etas(ideal_isometries[:_t], etas)
            _input_dim = 2 ** self._isometry_input_qubits[_t]
            _output_dim = 2 ** self._isometry_output_qubits[_t]

            def _cost_fn(_V: np.ndarray):
                _cost = 0
                for tip in etas.keys():
                    eta = etas[tip]
                    eta_t = _V @ eta @ _V.conj().transpose()
                    for povm_index in range(len(self._povm_operators[_t])):
                        op = np.kron(self._povm_operators[_t][povm_index],
                                     np.identity(2 ** self._ancilla_qubits[_t], dtype=complex))
                        init_state_index = tip[0]
                        cptp_index = tip[1]

                        measured_prob = mea_data[_t][(init_state_index, cptp_index, povm_index)]
                        recovered_prob = np.real(np.trace(op @ eta_t))
                        _cost += np.real(recovered_prob - measured_prob) ** 2
                return _cost

            def _euclidian_gradient(_V: np.ndarray):
                _G = np.zeros(_V.shape, dtype=complex)
                for tip in etas.keys():
                    eta = etas[tip]
                    eta_t = _V @ eta @ _V.conj().transpose()
                    _V_eta = _V @ eta

                    for povm_index in range(len(self._povm_operators[_t])):
                        # get measurement probability
                        init_state_index = tip[0]
                        cptp_index = tip[1]

                        measured_prob = mea_data[_t][(init_state_index, cptp_index, povm_index)]

                        # compute recovered probability
                        op = np.kron(self._povm_operators[_t][povm_index],
                                     np.identity(2 ** self._ancilla_qubits[_t], dtype=complex))
                        recovered_prob = np.real(np.trace(op @ eta_t))

                        # compute gradient w.r.t. the index
                        # -2 * (measured_prob - recovered_prob) * op @ _V @ eta
                        _g = -2 * (measured_prob - recovered_prob) * op @ _V_eta
                        _G += _g
                return _G

            cost_val = _cost_fn(ideal_isometries[_t])
            cost_vals.append(cost_val)
        return cost_vals


