from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Set, Tuple
import cfpq_data
import networkx as nx
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    State,
)


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
