from typing import Dict, List, Set, Tuple
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
