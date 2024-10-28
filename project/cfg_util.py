from dataclasses import dataclass
from typing import Dict, Set, Tuple, TypeAlias
from pyformlang.finite_automaton import Symbol, State
from scipy.sparse import csc_array
import pyformlang.cfg as pycfg
import networkx as nx

from project.adjacency_matrix_fa import AdjacencyMatrixFA
from project.graph_util import graph_to_nfa

LABEL_NAME = "label"
var_t: TypeAlias = pycfg.Variable
term_t: TypeAlias = pycfg.Terminal


def cfg_to_weak_normal_form(cfg: pycfg.CFG) -> pycfg.CFG:
    epsil_prod = cfg.get_nullable_symbols()
    cfg = cfg.to_normal_form()
    cfg_prod = list(cfg.productions)

    for symb in epsil_prod:
        cfg_prod.append(
            pycfg.Production(head=symb.value, body=[pycfg.Epsilon()], filtering=False)
        )

    return pycfg.CFG(start_symbol=cfg.start_symbol, productions=cfg_prod)


class CFGSolverTemaplate:
    @staticmethod
    def add_to_dict(dict, key, val):
        ts = dict.get(key, set())
        ts.add(val)
        dict[key] = ts

    def rebuild_graph(self, graph: nx.DiGraph) -> nx.MultiDiGraph:
        nodes = graph.nodes()
        edges = graph.edges(data=True)
        new_edges = []

        for edg in edges:
            d: Dict = edg[2]
            label_val = term_t(d.get(LABEL_NAME))

            var_set = self.terminal2Vars.get(label_val, set())
            for var in var_set:
                new_edges.append((edg[0], edg[1], {LABEL_NAME: var}))

        for var in self.epsilonVars:
            for node in nodes:
                new_edges.append((node, node, {LABEL_NAME: var}))

        res_graph = nx.MultiDiGraph(new_edges)
        for node in nodes:
            res_graph.add_node(node)

        return res_graph

    def process_cfg(self, cfg: pycfg.CFG) -> pycfg.CFG:
        cfg = cfg_to_weak_normal_form(cfg)

        for prod in cfg.productions:
            head = prod.head
            b = prod.body
            if len(b) == 1:
                if b[0] == pycfg.Epsilon():
                    self.epsilonVars.add(head)
                else:
                    self.add_to_dict(self.terminal2Vars, b[0], head)
            elif len(b) == 2:
                self.add_to_dict(self.vars2Vars, (b[0], b[1]), head)
            else:
                raise ValueError("Get normal cfg with len of product > 2")

        return cfg

    def __init__(self, cfg: pycfg.CFG, graph: nx.DiGraph):
        # N_i -> epsilon
        self.epsilonVars: Set[var_t] = set()

        # N_i -> t_j
        self.terminal2Vars: Dict[term_t, Set[var_t]] = {}

        # N_i -> N_j N_k
        self.vars2Vars: Dict[Tuple[var_t, var_t], Set[var_t]] = {}


def matrix_cfpq_solver_scope():
    dict_ind_t: TypeAlias = var_t | Symbol

    @dataclass(frozen=True, eq=True)
    class ProductionCase:
        head: var_t
        b1: var_t
        b2: var_t

    class MatrixCFPQSolver(CFGSolverTemaplate):
        def __init__(self, cfg: pycfg.CFG, graph: nx.DiGraph):
            super().__init__(cfg, graph)

            # N_i -> N_j N_k
            # (N_i, N_k) \in var2dependent[N_j]
            # (N_i, N_j) \in var2dependent[N_k]
            self.var2dependent: Dict[dict_ind_t, Set[ProductionCase]] = {}

            cfg = self.process_cfg(cfg)
            self.start_var = cfg.start_symbol

            for body in self.vars2Vars:
                heads = self.vars2Vars[body]
                for head in heads:
                    self.add_to_dict(
                        self.var2dependent,
                        body[0],
                        ProductionCase(head, body[0], body[1]),
                    )
                    self.add_to_dict(
                        self.var2dependent,
                        body[1],
                        ProductionCase(head, body[0], body[1]),
                    )

            new_graph = self.rebuild_graph(graph)
            nfa = graph_to_nfa(new_graph)

            self.fa_matrix = AdjacencyMatrixFA(nfa)
            self.bool_decomp: Dict[dict_ind_t, csc_array] = self.fa_matrix.bool_decomp

            shape = (self.fa_matrix.states_count, self.fa_matrix.states_count)
            for v in cfg.variables:
                if self.bool_decomp.get(v, None) is None:
                    self.bool_decomp[v] = csc_array(shape, dtype=bool)

        def step(self, pc: ProductionCase) -> bool:
            m1 = self.bool_decomp[pc.b1]
            m2 = self.bool_decomp[pc.b2]

            prev_c = self.bool_decomp[pc.head].count_nonzero()
            self.bool_decomp[pc.head] += m1 @ m2
            new_c = self.bool_decomp[pc.head].count_nonzero()
            return new_c != prev_c

        def count_matrixes(self):
            work_set: Set[ProductionCase] = set()

            for k in self.var2dependent:
                st = self.var2dependent[k]
                for pc in st:
                    work_set.add(pc)

            while work_set != set():
                pc = work_set.pop()
                is_changed = self.step(pc)
                if is_changed:
                    for npc in self.var2dependent.get(pc.head, set()):
                        work_set.add(npc)

        def solve_reach(
            self,
            from_nodes: Set[int],
            to_nodes: Set[int],
        ) -> Set[Tuple[int, int]]:
            self.count_matrixes()
            res = set()
            bd = self.bool_decomp[self.start_var]

            for i in from_nodes:
                for j in to_nodes:
                    from_ind = self.fa_matrix.state_to_index[State(i)]
                    to_ind = self.fa_matrix.state_to_index[State(j)]
                    if bd[from_ind, to_ind]:
                        res.add((i, j))

            return res

    return MatrixCFPQSolver


MatrixCFPQSolver = matrix_cfpq_solver_scope()


def matrix_based_cfpq(
    cfg: pycfg.CFG,
    graph: nx.DiGraph,
    start_nodes: Set[int] | None = None,
    final_nodes: Set[int] | None = None,
) -> Set[Tuple[int, int]]:
    if (start_nodes is None) or (start_nodes == set()):
        start_nodes = set(graph.nodes())

    if (final_nodes is None) or (final_nodes == set()):
        final_nodes = set(graph.nodes())

    s = MatrixCFPQSolver(cfg, graph)
    return s.solve_reach(start_nodes, final_nodes)


class HellingsCFPQSolver(CFGSolverTemaplate):
    def __init__(self, cfg: pycfg.CFG, graph: nx.DiGraph):
        super().__init__(cfg, graph)
        cfg = self.process_cfg(cfg)
        self.start_var = cfg.start_symbol

        graph = self.rebuild_graph(graph)
        self.unprocessed_reach = set(graph.edges(data=LABEL_NAME))
        self.result_reach = set()

    def step(self, from_to_var):
        if from_to_var not in self.result_reach:
            self.result_reach.add(from_to_var)

            for r_from_to_var in self.result_reach:
                if from_to_var[1] == r_from_to_var[0]:
                    vars = self.vars2Vars.get((from_to_var[2], r_from_to_var[2]), set())
                    for v in vars:
                        self.unprocessed_reach.add(
                            (from_to_var[0], r_from_to_var[1], v)
                        )

                if r_from_to_var[1] == from_to_var[0]:
                    vars = self.vars2Vars.get((r_from_to_var[2], from_to_var[2]), set())
                    for v in vars:
                        self.unprocessed_reach.add(
                            (r_from_to_var[0], from_to_var[1], v)
                        )

    def update_reach(self):
        while self.unprocessed_reach != set():
            self.step(self.unprocessed_reach.pop())

    def solve_reach(
        self,
        from_nodes: Set[int],
        to_nodes: Set[int],
    ) -> Set[Tuple[int, int]]:
        self.update_reach()
        res = set()

        for triple in self.result_reach:
            if triple[2] == self.start_var:
                if (triple[0] in from_nodes) and (triple[1] in to_nodes):
                    res.add((triple[0], triple[1]))

        return res


def hellings_based_cfpq(
    cfg: pycfg.CFG,
    graph: nx.DiGraph,
    start_nodes: Set[int] | None = None,
    final_nodes: Set[int] | None = None,
) -> Set[Tuple[int, int]]:
    if (start_nodes is None) or (start_nodes == set()):
        start_nodes = set(graph.nodes())

    if (final_nodes is None) or (final_nodes == set()):
        final_nodes = set(graph.nodes())

    s = HellingsCFPQSolver(cfg, graph)
    res = s.solve_reach(start_nodes, final_nodes)

    return res
