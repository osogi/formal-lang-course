import pytest
from project.graph_util import get_graph_data


def test_bzip_graph():
    expected_nodes_count = 632
    expected_edges_count = 556
    expected_labels = {"d", "a"}

    dt = get_graph_data("bzip")

    assert dt.nodes_count == expected_nodes_count
    assert dt.edges_count == expected_edges_count
    assert set(dt.labels) == expected_labels


def test_nonexist_graph():
    with pytest.raises(FileNotFoundError):
        get_graph_data("pikapikachu")
