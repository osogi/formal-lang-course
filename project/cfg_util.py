from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, TypeAlias
from pyformlang.finite_automaton import Symbol, State, DeterministicFiniteAutomaton
from pyformlang import rsa
from scipy import sparse
from scipy.sparse import csc_array
import pyformlang.cfg as pycfg
import networkx as nx

from project.adjacency_matrix_fa import (
    AdjacencyMatrixFA,
    intersect_automata,
)
from project.graph_util import graph_to_nfa
from project.rsm_util import get_symbol_spec_states, rsm_to_nfa

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


def tensor_based_cfpq(
    rsm: rsa.RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: Set[int] | None = None,
    final_nodes: Set[int] | None = None,
) -> Set[Tuple[int, int]]:
    symbol_to_spec_states = get_symbol_spec_states(rsm)
    rsm_fa = rsm_to_nfa(rsm)
    rsm_mfa = AdjacencyMatrixFA(rsm_fa)
    rsm_bd = rsm_mfa.bool_decomp

    graph = nx.MultiDiGraph(graph)
    graph_fa = graph_to_nfa(graph, start_nodes, final_nodes)
    graph_mfa = AdjacencyMatrixFA(graph_fa)
    graph_bd = graph_mfa.bool_decomp
    for symb in symbol_to_spec_states:
        if graph_bd.get(symb, None) is None:
            graph_bd[symb] = csc_array(
                (graph_mfa.states_count, graph_mfa.states_count), dtype=bool
            )
        if rsm_bd.get(symb, None) is None:
            rsm_bd[symb] = csc_array(
                (rsm_mfa.states_count, rsm_mfa.states_count), dtype=bool
            )

    prev_no_null_c = 0
    cur_no_null_c = -1
    index_to_state = {}

    def delta(tc: csc_array) -> Dict[Symbol, csc_array]:
        res_dict = {}
        buf_dict: Dict[Symbol, Tuple[List, List]] = {}
        from_inds, to_inds = tc.nonzero()

        def add_to_buf_dict(key, a, b):
            arrs = buf_dict.get(key)
            if arrs is None:
                buf_dict[key] = ([], [])
                arrs = buf_dict[key]
            arr1, arr2 = arrs
            arr1.append(a)
            arr2.append(b)

        for ind in range(len(from_inds)):
            from_ind = from_inds[ind]
            to_ind = to_inds[ind]

            from_st = index_to_state[from_ind]
            to_st = index_to_state[to_ind]

            (from_symb, from_rsm_st), from_graph_st = from_st.value
            (to_symb, to_rsm_st), to_graph_st = to_st.value
            if from_symb != to_symb:
                raise ValueError("expected the same symbols")

            symb = from_symb
            ss_set, fs_set = symbol_to_spec_states[symb]

            from_graph_ind = graph_mfa.state_to_index[from_graph_st]
            to_graph_ind = graph_mfa.state_to_index[to_graph_st]

            if (from_rsm_st in ss_set) and (to_rsm_st in fs_set):
                add_to_buf_dict(symb, from_graph_ind, to_graph_ind)

        # restore matrixes from arrays
        for symb in buf_dict:
            row_ind, col_ind = buf_dict[symb]
            data = [True] * len(row_ind)

            res_dict[symb] = csc_array(
                (data, (row_ind, col_ind)),
                shape=(graph_mfa.states_count, graph_mfa.states_count),
                dtype=bool,
            )

        return res_dict

    delta_bd = None
    while prev_no_null_c != cur_no_null_c:
        prev_no_null_c = cur_no_null_c

        if delta_bd is None:
            buf_mfa = intersect_automata(rsm_mfa, graph_mfa)
            for k in buf_mfa.state_to_index:
                v = buf_mfa.state_to_index[k]
                index_to_state[v] = k
        else:
            for symb in delta_bd:
                buf_mfa.bool_decomp[symb] += sparse.kron(
                    rsm_mfa.bool_decomp[symb], delta_bd[symb], "csc"
                )

        tc = buf_mfa.full_transitive_closure()
        delta_bd = delta(tc)

        cur_no_null_c = 0
        for symb in delta_bd:
            d = delta_bd[symb]
            graph_bd[symb] += d
            cur_no_null_c += graph_bd[symb].count_nonzero()

    res = set()
    for ss in graph_mfa.st_states:
        for fs in graph_mfa.fin_states:
            from_ind = graph_mfa.state_to_index[ss]
            to_ind = graph_mfa.state_to_index[fs]
            if graph_bd[rsm.initial_label][from_ind, to_ind]:
                res.add((ss, fs))

    return res


@dataclass(frozen=True)
class RsmState:
    var: Symbol
    sub_state: str


@dataclass(frozen=True)
class SPPFNode:
    gssn: GSSNode
    state: RsmState
    node: int


class GSSNode:
    state: RsmState
    node: int
    edges: Dict[RsmState, Set[GSSNode]]
    pop_set: Set[int]

    def __init__(self, st: RsmState, nd: int):
        self.state = st
        self.node = nd
        self.edges = {}
        self.pop_set = set()

    def pop(self, cur_node: int) -> Set[SPPFNode]:
        res_set = set()

        if cur_node not in self.pop_set:
            for new_st in self.edges:
                gses = self.edges[new_st]
                for gs in gses:
                    res_set.add(SPPFNode(gs, new_st, cur_node))

            self.pop_set.add(cur_node)
        return res_set

    def add_edge(self, ret_st: RsmState, ptr: GSSNode) -> Set[SPPFNode]:
        res_set = set()

        st_edges = self.edges.get(ret_st, set())
        if ptr not in st_edges:
            st_edges.add(ptr)
            for cur_node in self.pop_set:
                res_set.add(SPPFNode(ptr, ret_st, cur_node))

        self.edges[ret_st] = st_edges

        return res_set


class GSStack:
    body: Dict[Tuple[RsmState, int], GSSNode]

    def __init__(self):
        self.body = {}

    def get_node(self, rsm_st: RsmState, node: int):
        res = self.body.get((rsm_st, node), None)
        if res is None:
            res = GSSNode(rsm_st, node)
            self.body[(rsm_st, node)] = res
        return res


@dataclass
class RsmStateData:
    term_edges: Dict[Symbol, RsmState]
    var_edges: Dict[Symbol, Tuple[RsmState, RsmState]]
    # first RsmState start-state of symbol
    # second RsmState next stete after var (return state)

    is_final: bool


class GllCFPQSolver:
    def is_term(self, s: str) -> bool:
        return Symbol(s) not in self.rsmstate2data

    def init_graph_data(self, graph: nx.DiGraph):
        # Init data for graph traverse
        edges = graph.edges(data=LABEL_NAME)

        for n in graph.nodes():
            self.nodes2edges[n] = {}

        for from_n, to_n, symb in edges:
            if symb is not None:
                edges = self.nodes2edges[from_n]
                s: Set = edges.get(symb, set())
                s.add(to_n)
                edges[symb] = s

    def init_rsm_data(self, rsm: rsa.RecursiveAutomaton):
        # Init data for RSM traverse
        for var in rsm.boxes:
            self.rsmstate2data[var] = {}

        for var in rsm.boxes:
            box = rsm.boxes[var]
            fa: DeterministicFiniteAutomaton = box.dfa
            gbox = fa.to_networkx()

            sub_dict = self.rsmstate2data[var]

            for sub_state in gbox.nodes:
                is_fin = sub_state in fa.final_states
                sub_dict[sub_state] = RsmStateData({}, {}, is_fin)

            edges = gbox.edges(data=LABEL_NAME)
            for from_st, to_st, symb in edges:
                if symb is not None:
                    st_edges = sub_dict[from_st]
                    if self.is_term(symb):
                        st_edges.term_edges[symb] = RsmState(var, to_st)
                    else:
                        bfa: DeterministicFiniteAutomaton = rsm.boxes[Symbol(symb)].dfa
                        box_start = bfa.start_state.value
                        st_edges.var_edges[symb] = (
                            RsmState(Symbol(symb), box_start),
                            RsmState(var, to_st),
                        )

        start_symb = rsm.initial_label
        start_fa: DeterministicFiniteAutomaton = rsm.boxes[start_symb].dfa
        self.start_rstate = RsmState(start_symb, start_fa.start_state.value)

    def __init__(
        self,
        rsm: rsa.RecursiveAutomaton,
        graph: nx.DiGraph,
    ):
        self.nodes2edges: Dict[int, Dict[Symbol, Set[int]]] = {}
        self.rsmstate2data: Dict[Symbol, Dict[str, RsmStateData]] = {}
        self.start_rstate: RsmState

        self.rsm = rsm
        self.graph = graph

        self.init_graph_data(graph)
        self.init_rsm_data(rsm)

        self.gss = GSStack()
        self.accept_gssnode = self.gss.get_node(RsmState(Symbol("$"), "fin"), -1)

        self.unprocessed: Set[SPPFNode] = set()
        self.added: Set[SPPFNode] = set()

    def add_sppf_nodes(self, snodes: Set[SPPFNode]):
        snodes.difference_update(self.added)

        self.added.update(snodes)
        self.unprocessed.update(snodes)

    def filter_poped_nodes(
        self, snodes: Set[SPPFNode], prev_snode: SPPFNode
    ) -> Tuple[Set[SPPFNode], Set[Tuple[int, int]]]:
        node_res_set = set()
        start_fin_res_set = set()

        for sn in snodes:
            if sn.gssn == self.accept_gssnode:
                start_node = prev_snode.gssn.node
                fin_node = sn.node
                start_fin_res_set.add((start_node, fin_node))
            else:
                node_res_set.add(sn)

        return (node_res_set, start_fin_res_set)

    def step(self, sppfnode: SPPFNode) -> Set[Tuple[int, int]]:
        rsm_st = sppfnode.state
        rsm_dat = self.rsmstate2data[rsm_st.var][rsm_st.sub_state]
        # print(f"RSM dat: {rsm_dat}")

        def term_step():
            rsm_terms = rsm_dat.term_edges
            graph_terms = self.nodes2edges[sppfnode.node]
            for term in rsm_terms:
                if term in graph_terms:
                    new_sppf_nodes = set()
                    rsm_new_st = rsm_terms[term]
                    graph_new_nodes = graph_terms[term]
                    for gn in graph_new_nodes:
                        new_sppf_nodes.add(SPPFNode(sppfnode.gssn, rsm_new_st, gn))

                    self.add_sppf_nodes(new_sppf_nodes)

        def var_step() -> Set[Tuple[int, int]]:
            start_fin_set = set()
            for var in rsm_dat.var_edges:
                var_start_rsm_st, ret_rsm_st = rsm_dat.var_edges[var]

                inner_gss_node = self.gss.get_node(var_start_rsm_st, sppfnode.node)
                post_pop_sppf_nodes = inner_gss_node.add_edge(ret_rsm_st, sppfnode.gssn)

                post_pop_sppf_nodes, sub_start_fin_set = self.filter_poped_nodes(
                    post_pop_sppf_nodes, sppfnode
                )

                self.add_sppf_nodes(post_pop_sppf_nodes)
                self.add_sppf_nodes(
                    set([SPPFNode(inner_gss_node, var_start_rsm_st, sppfnode.node)])
                )

                start_fin_set.update(sub_start_fin_set)

            return start_fin_set

        def pop_step() -> Set[Tuple[int, int]]:
            new_sppf_nodes = sppfnode.gssn.pop(sppfnode.node)
            new_sppf_nodes, start_fin_set = self.filter_poped_nodes(
                new_sppf_nodes, sppfnode
            )
            self.add_sppf_nodes(new_sppf_nodes)
            return start_fin_set

        term_step()
        res_set = var_step()

        if rsm_dat.is_final:
            res_set.update(pop_step())

        return res_set

    def solve_reach(
        self,
        from_nodes: Set[int],
        to_nodes: Set[int],
    ) -> Set[Tuple[int, int]]:
        reach_set = set()
        for snode in from_nodes:
            gssn = self.gss.get_node(self.start_rstate, snode)
            gssn.add_edge(RsmState(Symbol("$"), "fin"), self.accept_gssnode)

            self.add_sppf_nodes(set([SPPFNode(gssn, self.start_rstate, snode)]))

        while self.unprocessed != set():
            reach_set.update(self.step(self.unprocessed.pop()))

        filtered_set = set()
        for st_fin in reach_set:
            if st_fin[1] in to_nodes:
                filtered_set.add(st_fin)
        return filtered_set


def gll_based_cfpq(
    rsm: rsa.RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: Set[int] | None = None,
    final_nodes: Set[int] | None = None,
) -> Set[Tuple[int, int]]:
    if (start_nodes is None) or (start_nodes == set()):
        start_nodes = set(graph.nodes())
    if (final_nodes is None) or (final_nodes == set()):
        final_nodes = set(graph.nodes())

    s = GllCFPQSolver(rsm, graph)
    return s.solve_reach(start_nodes, final_nodes)
