from typing import Dict, Iterable, List, Set, Tuple
import numpy
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol
from scipy import sparse


class AdjacencyMatrixFA:
    def __init__(self, fa: NondeterministicFiniteAutomaton) -> None:
        symbols: Set[Symbol] = fa.symbols
        states: Set[State] = fa.states

        self.st_states: Set[State] = fa.start_states
        self.fin_states: Set[State] = fa.final_states
        self.states_count = len(states)
        self.state_to_index = dict(zip(states, range(self.states_count)))

        buf_bool_decomp: Dict[Symbol, Tuple[List[int], List[int]]] = {}

        for symb in symbols:
            buf_bool_decomp[symb] = ([], [])

        nx_graph = fa.to_networkx()
        edges = nx_graph.edges(data="label")

        for edge in edges:
            src = edge[0]
            dest = edge[1]
            symb = edge[2]

            if symb is not None:
                srcs, dests = buf_bool_decomp[symb]
                srcs.append(self.state_to_index[src])
                dests.append(self.state_to_index[dest])

        self.bool_decomp: Dict[Symbol, sparse.csc_array] = {}

        for symb in symbols:
            lists = buf_bool_decomp[symb]
            self.bool_decomp[symb] = sparse.csc_array(
                ([True] * len(lists[0]), lists),
                shape=(self.states_count, self.states_count),
                dtype=bool,
            )

    def create_start_vector(self) -> sparse.csc_array:
        st_ind = [self.state_to_index[st] for st in self.st_states]
        data = [True] * len(st_ind)
        zeroes = [0] * len(st_ind)

        vec = sparse.csc_array(
            (data, (zeroes, st_ind)),
            shape=(1, self.states_count),
            dtype=bool,
        )

        return vec

    def accepts(self, word: Iterable[Symbol]) -> bool:
        vec = self.create_start_vector()

        for symb in word:
            matrix = self.bool_decomp.get(symb)
            if matrix is not None:
                vec = vec @ matrix
            else:
                return False

        for fin_st in self.fin_states:
            ind = self.state_to_index[fin_st]
            if vec[0, ind]:
                return True

        return False

    def is_empty(self) -> bool:
        matrix: numpy.ndarray | None = None

        for symb in self.bool_decomp.keys():
            if matrix is None:
                matrix = self.bool_decomp[symb].toarray()
            else:
                matrix += self.bool_decomp[symb]
        if matrix is None:
            return True

        for _ in range(self.states_count):
            matrix += matrix @ matrix

        for start in self.st_states:
            for fin in self.fin_states:
                if matrix[self.state_to_index[start], self.state_to_index[fin]]:
                    return False

        return True
