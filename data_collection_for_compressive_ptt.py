import copy
import itertools

import numpy as np
from typing import List, Tuple, Union
import pickle

from utils.pt_isometry import random_isometry, \
    get_prob_with_independent_intput_state, \
    isometries_to_choi_state

from tqdm import tqdm


def data_collect(isometries: List[np.ndarray], input_states: List[List[np.ndarray]],
                 povm_operators: List[List[np.ndarray]]):
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
    n_in_qlist = [round(np.log2(state_set[0].shape[0])) for state_set in input_states]
    n_out_qlist = [round(np.log2(povm_set[0].shape[0])) for povm_set in povm_operators]

    in_dim = [4 ** q for q in n_in_qlist]
    out_dim = [4 ** q for q in n_out_qlist]

    total_pieces = int(np.prod(in_dim) * np.prod(out_dim))
    pbar = tqdm(
        total=total_pieces,
        delay=3
    )

    collected_data = []
    for k in range(n_time_steps):
        input_state_indexes = list(itertools.product(*tuple(list(range(4 ** n_in_qlist[_i])) for _i in range(k + 1))))
        povm_op_indexes = list(itertools.product(*tuple(list(range(4 ** n_out_qlist[_i])) for _i in range(k + 1))))
        single_k_data = {}
        for state_index in input_state_indexes:
            for povm_index in povm_op_indexes:
                in_state = [input_states[_t][state_index[_t]] for _t in range(len(state_index))]
                povm_op = [povm_operators[_t][povm_index[_t]] for _t in range(len(state_index))]
                prob = get_prob_with_independent_intput_state(isometries[:len(state_index)], in_state, povm_op)
                single_k_data[(state_index, povm_index)] = prob
                pbar.update(1)

        collected_data.append(single_k_data)

    pbar.close()

    return collected_data


def complete_data_collect(isometries: List[np.ndarray], input_states: List[List[np.ndarray]],
                          povm_operators: List[List[np.ndarray]]):
    """
    Collect complete probability data with informationally complete input states and POVM operators.
    :param isometries: the isometry represented process tensor. A list of isometries (np.ndarray).
    :param input_states: a list of informationally complete input states.
    :param povm_operators: a list of informationally complete POVM operators.
    :return: probability data, with structure
                [data_with_max_time_step_0, data_with_max_time_step_1, ..., data_with_max_time_step_N-1]
                data_with_max_time_step_k: [(input_state_index, output_state_index, prob)...]
    """
    n_time_steps = len(isometries)
    n_in_qlist = [round(np.log2(state_set[0].shape[0])) for state_set in input_states]
    n_out_qlist = [round(np.log2(povm_set[0].shape[0])) for povm_set in povm_operators]

    total_pieces = np.prod(n_in_qlist) * np.prod(n_out_qlist)

    proc_tensor_choi = isometries_to_choi_state(isometries, n_in_qlist, n_out_qlist)

    collected_data = []

    input_state_indexes = list(
        itertools.product(*tuple(list(range(4 ** n_in_qlist[_i])) for _i in range(n_time_steps))))
    povm_op_indexes = list(itertools.product(*tuple(list(range(4 ** n_out_qlist[_i])) for _i in range(n_time_steps))))
    for state_index in input_state_indexes:
        for povm_index in povm_op_indexes:
            in_state = [input_states[_t][state_index[_t]] for _t in range(len(state_index))]
            povm_op = [povm_operators[_t][povm_index[_t]] for _t in range(len(state_index))]
            prob = get_prob_with_independent_intput_state(isometries[:len(state_index)], in_state, povm_op)
            collected_data.append((state_index, povm_index, prob))

    return collected_data, proc_tensor_choi


def save_data(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_data(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

