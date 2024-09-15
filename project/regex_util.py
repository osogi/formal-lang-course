from pyformlang.finite_automaton import DeterministicFiniteAutomaton
from pyformlang.regular_expression import Regex


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    regex_c = Regex(regex)
    nfa = regex_c.to_epsilon_nfa()
    if nfa is not None:
        dfa: DeterministicFiniteAutomaton = nfa.to_deterministic()
        return dfa.minimize()
    else:
        raise TypeError("NFA generated from the target regex must not be None")
