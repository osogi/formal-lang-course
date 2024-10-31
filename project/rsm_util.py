from typing import Dict, Set, Tuple
from pyformlang import rsa, cfg as pycfg
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol, State

LABEL_NAME = "label"


def ebnf_to_rsm(ebnf: str) -> rsa.RecursiveAutomaton:
    return rsa.RecursiveAutomaton.from_text(ebnf)


def cfg_to_rsm(cfg: pycfg.CFG) -> rsa.RecursiveAutomaton:
    return ebnf_to_rsm(cfg.to_text())


def rsm_to_nfa(rsm: rsa.RecursiveAutomaton) -> NondeterministicFiniteAutomaton:
    transitions = []
    start_states = set()
    fin_states = set()

    boxes: Dict[Symbol, rsa.Box] = rsm.boxes
    for k in boxes:
        v = boxes[k]
        fa = v.dfa

        def new_state(s):
            return State((k, s))

        ss = set([new_state(s.value) for s in fa.start_states])
        start_states = start_states.union(ss)

        fs = set([new_state(s.value) for s in fa.final_states])
        fin_states = fin_states.union(fs)

        trs = fa.to_networkx().edges(data=LABEL_NAME)
        for t in trs:
            transitions.append((new_state(t[0]), t[2], new_state(t[1])))

    res: NondeterministicFiniteAutomaton = NondeterministicFiniteAutomaton(
        start_state=start_states, final_states=fin_states
    )
    res.add_transitions(transitions)

    return res


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
