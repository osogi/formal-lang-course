from typing import Dict, Iterable, List, Set, Tuple
import numpy
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol
from scipy import sparse


class AdjacencyMatrixFA:
    def _construct(
        self,
        bool_decomp: Dict[Symbol, sparse.csc_array],
        start_states: Set[State],
        final_states: Set[State],
        state_to_index: dict[State, int],
    ):
        self.st_states: Set[State] = start_states
        self.fin_states: Set[State] = final_states
        self.states_count = len(state_to_index)
        self.state_to_index = state_to_index

        # TODO: maybe should be changed to function, but i am lazy
        self.bool_decomp = bool_decomp

    def _from_nfa(self, fa: NondeterministicFiniteAutomaton):
        symbols: Set[Symbol] = fa.symbols
        states: Set[State] = fa.states
        st_states: Set[State] = fa.start_states
        fin_states: Set[State] = fa.final_states
        states_count = len(states)
        state_to_index = dict(zip(states, range(states_count)))

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
                srcs.append(state_to_index[src])
                dests.append(state_to_index[dest])

        bool_decomp: Dict[Symbol, sparse.csc_array] = {}

        for symb in symbols:
            lists = buf_bool_decomp[symb]
            bool_decomp[symb] = sparse.csc_array(
                ([True] * len(lists[0]), lists),
                shape=(states_count, states_count),
                dtype=bool,
            )

        self._construct(bool_decomp, st_states, fin_states, state_to_index)

    def __init__(
        self,
        arg: (
            NondeterministicFiniteAutomaton
            | Tuple[
                Dict[Symbol, sparse.csc_array], Set[State], Set[State], dict[State, int]
            ]
        ),
    ) -> None:
        if isinstance(arg, NondeterministicFiniteAutomaton):
            self._from_nfa(arg)
        else:
            self._construct(*arg)

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

    def full_transitive_closure(self) -> numpy.ndarray | None:
        matrix: numpy.ndarray | None = None

        for symb in self.bool_decomp.keys():
            if matrix is None:
                matrix = self.bool_decomp[symb].toarray()
            else:
                matrix += self.bool_decomp[symb]
        if matrix is None:
            matrix = numpy.zeros((self.states_count, self.states_count), bool)

        for i in range(matrix.shape[0]):
            matrix[i, i] = True

        # TODO: it can be optimized
        for _ in range(self.states_count):
            matrix = matrix @ matrix

        return matrix

    def is_empty(self) -> bool:
        matrix = self.full_transitive_closure()
        if matrix is None:
            return True

        for start in self.st_states:
            for fin in self.fin_states:
                if matrix[self.state_to_index[start], self.state_to_index[fin]]:
                    return False

        return True


def tensor_dot(m1: sparse.csc_array, m2: sparse.csc_array) -> sparse.csc_array:
    row_ind: List[int] = []
    col_ind: List[int] = []

    for i1 in range(m1.shape[0]):
        for j1 in range(m1.shape[1]):
            for i2 in range(m2.shape[0]):
                for j2 in range(m2.shape[1]):
                    if (m1[i1, j1]) and (m2[i2, j2]):
                        row_ind.append(i1 + i2 * m1.shape[0])
                        col_ind.append(j1 + j2 * m1.shape[1])

    data = [True] * len(row_ind)
    return sparse.csc_array(
        (data, (row_ind, col_ind)),
        (m1.shape[0] * m2.shape[0], m1.shape[1] * m2.shape[1]),
    )


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    symbs1 = set(automaton1.bool_decomp.keys())
    symbs2 = set(automaton2.bool_decomp.keys())
    symbs = symbs1.intersection(symbs2)

    bool_decomp: Dict[Symbol, sparse.csc_array] = {}

    for symb in symbs:
        m1 = automaton1.bool_decomp[symb]
        m2 = automaton2.bool_decomp[symb]
        bool_decomp[symb] = tensor_dot(m1, m2)

    def state_intersection(st1: Set[State], st2: Set[State]) -> Set[State]:
        st: Set[State] = set()
        for s1 in st1:
            for s2 in st2:
                st.add(State((s1.value, s2.value)))
        return st

    start_states = state_intersection(automaton1.st_states, automaton2.st_states)
    final_states = state_intersection(automaton1.fin_states, automaton2.fin_states)
    state_to_index: Dict[State, int] = {}

    for s1 in automaton1.state_to_index.keys():
        for s2 in automaton2.state_to_index.keys():
            st = State((s1.value, s2.value))
            index = (
                automaton1.state_to_index[s1]
                + automaton2.state_to_index[s2] * automaton1.states_count
            )
            state_to_index[st] = index

    return AdjacencyMatrixFA((bool_decomp, start_states, final_states, state_to_index))
