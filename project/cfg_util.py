import pyformlang.cfg as pycfg


def cfg_to_weak_normal_form(cfg: pycfg.CFG) -> pycfg.CFG:
    epsil_prod = cfg.get_nullable_symbols()
    cfg = cfg.to_normal_form()
    cfg_prod = list(cfg.productions)

    for symb in epsil_prod:
        cfg_prod.append(
            pycfg.Production(head=symb.value, body=[pycfg.Epsilon()], filtering=False)
        )

    return pycfg.CFG(start_symbol=cfg.start_symbol, productions=cfg_prod)
