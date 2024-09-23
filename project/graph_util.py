from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Set, Tuple
import cfpq_data
import networkx as nx
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    State,
)

from project.regex_util import regex_to_dfa
from project.adjacency_matrix_fa import AdjacencyMatrixFA, intersect_automata


@dataclass
class GraphInfo:
    nodes_count: int
    edges_count: int
    labels: List[Any]


def get_graph_data(name: str) -> GraphInfo:
    path = cfpq_data.download(name)
    graph = cfpq_data.graph_from_csv(path)

    graph_data = GraphInfo(
        nodes_count=graph.number_of_nodes(),
        edges_count=graph.number_of_edges(),
        labels=cfpq_data.get_sorted_labels(graph),
    )

    return graph_data


def save_graph(graph: nx.MultiDiGraph, output_path: Path) -> None:
    pdt_graph = nx.drawing.nx_pydot.to_pydot(graph)
    pdt_graph.write_raw(output_path)


def save_labeled_two_cycles_graph(
    n: int, m: int, labels: Tuple[str, str], output_path: Path
) -> None:
    graph = cfpq_data.labeled_two_cycles_graph(n, m, labels=labels)
    save_graph(graph, output_path)


def graph_to_nfa(
    graph: nx.MultiDiGraph,
    start_states: Set[int] | None = None,
    final_states: Set[int] | None = None,
) -> NondeterministicFiniteAutomaton:
    all_nodes = set([int(i) for i in graph.nodes(data=False)])

    if (start_states is None) or (len(start_states) == 0):
        start_states = all_nodes
    if (final_states is None) or (len(final_states) == 0):
        final_states = all_nodes

    nfa = NondeterministicFiniteAutomaton.from_networkx(graph)

    for ss in start_states:
        nfa.add_start_state(State(ss))
    for fs in final_states:
        nfa.add_final_state(State(fs))

    return nfa.remove_epsilon_transitions()


def tensor_based_rpq(
    regex: str, graph: nx.MultiDiGraph, start_nodes: Set[int], final_nodes: set[int]
) -> Set[Tuple[int, int]]:
    regex_fa = regex_to_dfa(regex)
    graph_fa = graph_to_nfa(graph, start_nodes, final_nodes)

    regex_mfa = AdjacencyMatrixFA(regex_fa)
    graph_mfa = AdjacencyMatrixFA(graph_fa)
    res_mfa = intersect_automata(regex_mfa, graph_mfa)

    result_set: Set[Tuple[int, int]] = set()

    closure = res_mfa.full_transitive_closure()
    if closure is None:
        return result_set

    for st in res_mfa.st_states:
        for fin in res_mfa.fin_states:
            if closure[res_mfa.state_to_index[st], res_mfa.state_to_index[fin]]:
                result_set.add((st.value[1], fin.value[1]))

    return result_set
