import pytest
from tests.autotests.helper import generate_rnd_graph, generate_rnd_dense_graph
from networkx import MultiDiGraph
from tests.autotests.constants import LABELS
import random

funcs = [generate_rnd_dense_graph, generate_rnd_graph]


@pytest.fixture(scope="function", params=range(8))
def graph(request) -> MultiDiGraph:
    fun = random.choice(funcs)
    return fun(1, 100, LABELS)
