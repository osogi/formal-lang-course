import pyformlang.cfg as pycfg


def cfg_to_weak_normal_form(cfg: pycfg.CFG) -> pycfg.CFG:
    cfg = cfg.remove_useless_symbols()
    cfg_prod = cfg.productions
    cfg_prod = cfg._decompose_productions(cfg_prod)

    cfg = pycfg.CFG(start_symbol=cfg.start_symbol, productions=cfg_prod)
    cfg_prod = cfg._get_productions_with_only_single_terminals()

    wcfg = pycfg.CFG(start_symbol=cfg.start_symbol, productions=cfg_prod)
    return wcfg
