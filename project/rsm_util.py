from typing import Dict, Set, Tuple
from pyformlang import rsa, cfg as pycfg
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol, State
import networkx as nx

from project.graph_util import graph_to_nfa


def ebnf_to_rsm(ebnf: str) -> rsa.RecursiveAutomaton:
    return rsa.RecursiveAutomaton.from_text(ebnf)


def cfg_to_rsm(cfg: pycfg.CFG) -> rsa.RecursiveAutomaton:
    return ebnf_to_rsm(cfg.to_text())


def rsm_to_nfa(rsm: rsa.RecursiveAutomaton) -> NondeterministicFiniteAutomaton:
    graph = nx.MultiDiGraph()
    start_states = set()
    fin_states = set()

    boxes: Dict[Symbol, rsa.Box] = rsm.boxes
    for k in boxes:
        v = boxes[k]
        fa = v.dfa
        graph.update(fa.to_networkx())
        start_states = start_states.union(fa.start_states)
        fin_states = fin_states.union(fa.final_states)

    return graph_to_nfa(graph, start_states, fin_states)


def get_symbol_spec_states(
    rsm: rsa.RecursiveAutomaton,
) -> Dict[Symbol, Tuple[Set[State], Set[State]]]:
    res_dict = {}

    boxes: Dict[Symbol, rsa.Box] = rsm.boxes
    for k in boxes:
        v = boxes[k]
        fa = v.dfa
        res_dict[k] = (fa.start_states, fa.final_states)

    return res_dict
