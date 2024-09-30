import numpy
from scipy import sparse
from typing import Dict, Iterable, Set, Tuple
import networkx as nx
from pyformlang.finite_automaton import Symbol

from project.graph_util import graph_to_nfa
from project.regex_util import regex_to_dfa
from project.adjacency_matrix_fa import AdjacencyMatrixFA


def direct_sum_bool_decomp(
    bd1: Dict[Symbol, sparse.csc_array], bd2: Dict[Symbol, sparse.csc_array]
) -> Dict[Symbol, sparse.csc_array]:
    symbs1 = set(bd1.keys())
    symbs2 = set(bd2.keys())
    symbs = symbs1.intersection(symbs2)

    bool_decomp: Dict[Symbol, sparse.csc_array] = {}

    for symb in symbs:
        m1 = bd1[symb]
        m2 = bd2[symb]
        bool_decomp[symb] = sparse.block_diag((m1, m2), "csc")

    return bool_decomp


def init_front(
    dfa_start_state: int,
    dfa_state_count: int,
    nfa_sart_states: Iterable[int],
    nfa_state_count: int,
) -> sparse.csr_array:
    k = dfa_state_count
    n = nfa_state_count

    row_ind = []
    col_ind = []

    row_ind.append(dfa_start_state)
    col_ind.append(dfa_start_state)

    for ind in nfa_sart_states:
        row_ind.append(dfa_start_state)
        col_ind.append(k + ind)

    data = [True] * len(row_ind)

    front = sparse.csr_array((data, (row_ind, col_ind)), shape=(k, k + n))
    return front


def extract_part_by_statrt_state(
    matr: sparse.csr_array, k: int, start_state_num: int = 1
) -> sparse.csr_array:
    return matr[start_state_num * k : (start_state_num + 1) * k, :]  # noqa: E203


def add_paths_from_matrix(
    res_row_ind: list[int],
    res_col_ind: list[int],
    matr: sparse.csr_array,
    start_state_num: int = 0,
):
    k = matr.shape[0]
    nz_row, nz_col = matr.nonzero()

    l_inds = numpy.where(nz_col < k)[0]

    if len(l_inds) < 1:
        return

    m_rows: Set[int] = set()
    for m_ind in l_inds:
        m_rows.add(nz_row[m_ind])
        m_col = nz_col[m_ind]

    for i in range(len(nz_row)):
        row = nz_row[i]
        col = nz_col[i]

        if row in m_rows:  # maybe excessive
            res_row_ind.append(start_state_num * k + m_col)
            res_col_ind.append(col)


def move_front(
    f: sparse.csr_array,
    bool_decomp: Dict[Symbol, sparse.csc_array],
    start_state_count: int = 1,
) -> sparse.csr_array:
    k = f.shape[0] // start_state_count
    res_row_ind = []
    res_col_ind = []

    for symb in bool_decomp.keys():
        matr = bool_decomp[symb]

        r = f @ matr
        for i in range(start_state_count):
            add_paths_from_matrix(
                res_row_ind, res_col_ind, extract_part_by_statrt_state(r, k, i), i
            )

    data = [True] * len(res_row_ind)
    new_front = sparse.csr_array(
        (data, (res_row_ind, res_col_ind)), shape=f.shape, dtype=bool
    )

    return new_front


def csr_array_remove(src: sparse.csr_array, mask: sparse.csr_array) -> sparse.csr_array:
    # maybe it can be optimized
    src_row, src_col = src.nonzero()

    for i in range(len(src_row)):
        row = src_row[i]
        col = src_col[i]

        if mask[row, col]:
            src[row, col] = 0

    src.eliminate_zeros()
    return src


def ms_bfs_based_rpq(
    regex: str, graph: nx.MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> Set[Tuple[int, int]]:
    result_set: Set[Tuple[int, int]] = set()

    regex_dfa = regex_to_dfa(regex)
    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)

    regex_mfa = AdjacencyMatrixFA(regex_dfa)
    graph_mfa = AdjacencyMatrixFA(graph_nfa)

    reg_start_state = regex_mfa.state_to_index[list(regex_mfa.st_states)[0]]
    graph_start_states = [graph_mfa.state_to_index[st] for st in graph_mfa.st_states]

    k = regex_mfa.states_count
    n = graph_mfa.states_count

    f_list = []
    visited_list = []
    start_state_ind_to_num: Dict[int, int] = {}

    start_state_num = 0
    for st_ind in graph_start_states:
        buf_f = init_front(
            reg_start_state,
            k,
            {st_ind},
            n,
        )
        buf_visited: sparse.csr_array = buf_f[:, k:]
        f_list.append(buf_f)
        visited_list.append(buf_visited)

        start_state_ind_to_num[st_ind] = start_state_num
        start_state_num += 1

    f: sparse.csr_array = sparse.vstack(f_list, "csr")
    visited: sparse.csr_array = sparse.vstack(visited_list, "csr")
    start_state_count = len(graph_start_states)

    bd = direct_sum_bool_decomp(regex_mfa.bool_decomp, graph_mfa.bool_decomp)

    while (f[:, k:]).size > 0:
        f = move_front(f, bd, start_state_count=start_state_count)
        f[:, k:] = csr_array_remove(f[:, k:], visited)
        visited = visited + f[:, k:]

    reg_final_ind = [regex_mfa.state_to_index[st] for st in regex_mfa.fin_states]
    for start in graph_mfa.st_states:
        for final in graph_mfa.fin_states:
            start_ind = graph_mfa.state_to_index[start]
            final_ind = graph_mfa.state_to_index[final]
            start_num = start_state_ind_to_num[start_ind]

            buf_visited = extract_part_by_statrt_state(visited, k, start_num)
            for i in reg_final_ind:
                if buf_visited[i, final_ind]:
                    result_set.add((start.value, final.value))
                    break

    return result_set
