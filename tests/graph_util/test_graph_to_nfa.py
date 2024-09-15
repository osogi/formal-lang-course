import pytest  # noqa: F401
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton
from networkx import MultiDiGraph
import random
from tests.autotests.helper import GraphWordsHelper, generate_rnd_start_and_final
from tests.autotests.constants import IS_FINAL, IS_START
from project.graph_util import graph_to_nfa


class TestGraphToNfa:
    def test_not_specified_start(self, graph: MultiDiGraph) -> None:
        copy_graph = MultiDiGraph(graph.copy())
        _, final_nodes = generate_rnd_start_and_final(copy_graph)
        nfa: NondeterministicFiniteAutomaton = graph_to_nfa(
            copy_graph, set(), final_nodes.copy()
        )
        for _, data in copy_graph.nodes(data=True):
            data[IS_START] = True

        words_helper = GraphWordsHelper(copy_graph)
        words = words_helper.get_words_with_limiter(random.randint(10, 100))
        if len(words) == 0:
            assert nfa.is_empty()
        else:
            word = random.choice(words)
            assert nfa.accepts(word)

    def test_not_specified_final(self, graph: MultiDiGraph) -> None:
        copy_graph = MultiDiGraph(graph.copy())
        start_nodes, _ = generate_rnd_start_and_final(copy_graph)
        nfa: NondeterministicFiniteAutomaton = graph_to_nfa(copy_graph, start_nodes)

        for _, data in copy_graph.nodes(data=True):
            data[IS_FINAL] = True

        words_helper = GraphWordsHelper(copy_graph)
        words = words_helper.get_words_with_limiter(random.randint(10, 100))
        if len(words) == 0:
            assert nfa.is_empty()
        else:
            word = random.choice(words)
            assert nfa.accepts(word)
