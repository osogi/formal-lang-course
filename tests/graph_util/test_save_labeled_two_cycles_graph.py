import pytest
import cfpq_data
from pathlib import Path
import networkx
import pydot
from project.graph_util import save_labeled_two_cycles_graph


def test_saving(tmpdir):
    save_path = Path(tmpdir).joinpath("graph.dot")
    n = 13
    m = 42
    labels = ("aboba", "answer")

    expected_nodes_count = n + m + 1
    expected_labels = set(labels)

    save_labeled_two_cycles_graph(n, m, labels, save_path)

    load_graphs = pydot.graph_from_dot_file(save_path)

    if load_graphs is None:
        pytest.fail("Can't laod saved graph")

    load_graph = load_graphs[0]
    nx_graph: networkx.MultiDiGraph = networkx.drawing.nx_pydot.from_pydot(load_graph)

    assert nx_graph.number_of_nodes() == expected_nodes_count
    assert set(cfpq_data.get_sorted_labels(nx_graph)) == expected_labels
