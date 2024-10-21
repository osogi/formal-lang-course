from typing import Dict, Set, Tuple, TypeAlias
import pyformlang.cfg as pycfg
import networkx as nx

from project.adjacency_matrix_fa import AdjacencyMatrixFA
from project.graph_util import graph_to_nfa


def cfg_to_weak_normal_form(cfg: pycfg.CFG) -> pycfg.CFG:
    epsil_prod = cfg.get_nullable_symbols()
    cfg = cfg.to_normal_form()
    cfg_prod = list(cfg.productions)

    for symb in epsil_prod:
        cfg_prod.append(
            pycfg.Production(head=symb.value, body=[pycfg.Epsilon()], filtering=False)
        )

    return pycfg.CFG(start_symbol=cfg.start_symbol, productions=cfg_prod)


def hellings_solver_scope():
    label_name = "label"

    class SortedTuple(tuple):
        def __new__(cls, *args):
            return super().__new__(cls, sorted(args))

    var_t: TypeAlias = pycfg.Variable
    term_t: TypeAlias = pycfg.Terminal

    class HellingsSolver:
        def rebuild_graph(self, graph: nx.DiGraph) -> nx.MultiDiGraph:
            nodes = graph.nodes()
            edges = graph.edges(data=True)
            new_edges = []

            for edg in edges:
                d: Dict = edg[2]
                label_val = term_t(d.get(label_name))

                var_set = self.terminal2Vars.get(label_val, set())
                for var in var_set:
                    new_edges.append((edg[0], edg[1], {label_name: var}))

            for var in self.epsilonVars:
                for node in nodes:
                    new_edges.append((node, node, {label_name: var}))

            return nx.MultiDiGraph(new_edges)

        def _process_cfg(self, cfg: pycfg.CFG):
            cfg = cfg_to_weak_normal_form(cfg)

            def add_to_dict(dict, key, val):
                ts = dict.get(key, set())
                ts.add(val)
                dict[key] = ts

            for prod in cfg.productions:
                head = prod.head
                b = prod.body
                if len(b) == 1:
                    if b[0] == pycfg.Epsilon():
                        self.epsilonVars.add(head)
                    else:
                        add_to_dict(self.terminal2Vars, b[0], head)
                elif len(b) == 2:
                    add_to_dict(self.var2dependent, b[0], (head, b[1]))
                    add_to_dict(self.var2dependent, b[1], (head, b[0]))
                else:
                    raise ValueError("Get normal cfg with len of product > 2")

        def __init__(
            self,
            cfg: pycfg.CFG,
            graph: nx.DiGraph,
            start_nodes: Set[int] | None = None,
            final_nodes: Set[int] | None = None,
        ):
            # N_i -> epsilon
            self.epsilonVars: Set[var_t] = set()

            # N_i -> t_j
            self.terminal2Vars: Dict[term_t, Set[var_t]] = {}

            # N_i -> N_j N_k
            # (N_i, N_k) \in var2dependent[N_j]
            # (N_i, N_j) \in var2dependent[N_k]
            self.var2dependent: Dict[var_t, Set[Tuple[var_t, var_t]]] = {}

            self._process_cfg(cfg)
            new_graph = self.rebuild_graph(graph)
            nfa = graph_to_nfa(new_graph, start_nodes, final_nodes)

            fa_matrix = AdjacencyMatrixFA(nfa)
            self.bool_decomp = fa_matrix.bool_decomp

    return HellingsSolver


HellingsSolver = hellings_solver_scope()
