from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple
import cfpq_data
import networkx


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


def save_graph(graph: networkx.MultiDiGraph, output_path: Path) -> None:
    pdt_graph = networkx.drawing.nx_pydot.to_pydot(graph)
    pdt_graph.write_raw(output_path)


def save_labeled_two_cycles_graph(
    n: int, m: int, labels: Tuple[str, str], output_path: Path
) -> None:
    graph = cfpq_data.labeled_two_cycles_graph(n, m, labels=labels)
    save_graph(graph, output_path)
