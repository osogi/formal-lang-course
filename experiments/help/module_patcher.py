"""
Be careful when using this module, it was written for my one case only
And it is very VERY cringe
"""

import ast
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

MAIN_MOD = "project"
FUN_SUF = "_array"


class MatrixReplacer(ast.NodeTransformer):
    def __init__(self, f_old2new: Dict[str, str], s_old2new: Dict[str, str]):
        self.f_old2new = f_old2new
        self.s_old2new = s_old2new
        self.project_modules = []

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        old = node.attr
        new = self.f_old2new.get(old)
        if new is not None:
            if new != ("dok" + FUN_SUF):
                node.attr = new
        self.generic_visit(node)
        return node

    def visit_Call(self, node: ast.Call) -> Any:
        res_node = node
        func = node.func
        if isinstance(func, ast.Attribute):
            func_name = func.attr
            c_func = self.f_old2new.get(func_name)
            if c_func == ("dok" + FUN_SUF):
                new_attr = ast.Attribute(node, "todok", ast.Load())
                res_node = ast.Call(new_attr, [], [])
                ast.fix_missing_locations(res_node)
        self.generic_visit(node)
        return res_node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        mod = node.module
        if mod is not None:
            if mod.startswith(MAIN_MOD):
                self.project_modules.append(mod)
        self.generic_visit(node)
        return node

    def visit_Constant(self, node: ast.Constant) -> Any:
        const = node.value
        if isinstance(const, str):
            new = self.s_old2new.get(const)
            if new is not None:
                node.value = new
        self.generic_visit(node)
        return node


def module2file(module_name: str, prefix: str) -> Path:
    return Path(f"{prefix}{module_name.replace(".", "/")}.py")


def replace_matrix(
    p_module: Path, old2new: Dict[str, str], module: ModuleType | None = None
) -> ModuleType:
    source_code: str
    with open(p_module, "rt") as reader:
        source_code = reader.read()
    prefix = str(p_module).split(MAIN_MOD)[0]

    tree = ast.parse(source_code)

    f_old2new = {}
    for k in old2new.keys():
        v = old2new[k]
        f_old2new[k + FUN_SUF] = v + FUN_SUF

    patcher = MatrixReplacer(f_old2new, old2new)
    tree = patcher.visit(tree)
    code = compile(tree, p_module, "exec")

    if module is None:
        module = ModuleType(f"{p_module.name}-patched")
    exec(code, module.__dict__)

    for mod in patcher.project_modules:
        replace_matrix(Path(module2file(mod, prefix)), old2new, module)

    return module


# replace_matrix(Path("./project/adjacency_matrix_fa.py"), {"csc": "dok"})
