from typing import Dict, Iterable, List, Set, Tuple
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

    def full_transitive_closure(self) -> sparse.csc_array:
        col_row_ind = list(range(self.states_count))
        data = [True] * len(col_row_ind)

        matrix = sparse.csc_array((data, (col_row_ind, col_row_ind)))
        for symb in self.bool_decomp.keys():
            matrix += self.bool_decomp[symb]
        matrix = matrix_bin_power(matrix, self.states_count)

        return matrix

    def is_empty(self) -> bool:
        matrix = self.full_transitive_closure()

        for start in self.st_states:
            for fin in self.fin_states:
                if matrix[self.state_to_index[start], self.state_to_index[fin]]:
                    return False

        return True


def matrix_bin_power(m: sparse.csc_array, p: int) -> sparse.csc_array:
    result = m
    buf = m

    while p >= 1:
        if p % 2 == 1:
            result = result @ buf
        buf = buf @ buf
        p //= 2

    return result


def tensor_dot(m1: sparse.csc_array, m2: sparse.csc_array) -> sparse.csc_array:
    return sparse.kron(m1, m2, "csc")


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
                automaton1.state_to_index[s1] * automaton2.states_count
                + automaton2.state_to_index[s2]
            )
            state_to_index[st] = index

    return AdjacencyMatrixFA((bool_decomp, start_states, final_states, state_to_index))
