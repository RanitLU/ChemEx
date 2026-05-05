from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import root

from chemex.configuration.conditions import Conditions
from chemex.models.factory import model_factory
from chemex.parameters.setting import NameSetting, ParamLocalSetting
from chemex.parameters.userfunctions import user_function_registry
from chemex.typing import Array


NAME = "4st_binding_cyclic_cs_if"

TPL = ("temperature", "p_total", "l_total")


def calculate_residuals(
    concentrations: Array,
    p_total: float,
    l_total: float,
    kd_ab: float,
    kd_ac: float,
    keq_bd: float,
) -> Array:
    """
    A + L <-> B   conformational-selection binding branch
    A + L <-> C   induced-fit binding branch
    B     <-> D   bound-state rearrangement toward fully bound state

    The fourth edge C <-> D is thermodynamically implied by detailed balance.
    """
    a, l_free, b, c, d = concentrations

    return np.array(
        [
            l_total - (l_free + b + c + d),
            p_total - (a + b + c + d),
            kd_ab * b - a * l_free,
            kd_ac * c - a * l_free,
            keq_bd * b - d,
        ],
    )


@lru_cache(maxsize=100)
def calculate_concentrations(
    p_total: float,
    l_total: float,
    kd_ab: float,
    kd_ac: float,
    keq_bd: float,
) -> dict[str, float]:
    concentrations_start = (p_total, l_total, 0.0, 0.0, 0.0)

    results = root(
        calculate_residuals,
        concentrations_start,
        args=(p_total, l_total, kd_ab, kd_ac, keq_bd),
    )

    a, l_free, b, c, d = results["x"]

    return {"a": a, "l": l_free, "b": b, "c": c, "d": d}


def make_settings_4st_binding_cyclic_cs_if(
    conditions: Conditions,
) -> dict[str, ParamLocalSetting]:
    p_total = conditions.p_total
    l_total = conditions.l_total

    if p_total is None:
        msg = f"'p_total' must be specified to use the '{NAME}' model"
        raise ValueError(msg)

    if l_total is None:
        msg = f"'l_total' must be specified to use the '{NAME}' model"
        raise ValueError(msg)

    return {
        # Binding: apo A + ligand <-> B
        "kd_ab": ParamLocalSetting(
            name_setting=NameSetting("kd_ab", "", ("temperature",)),
            value=1e-6,
            min=0.0,
            max=1.0,
            vary=True,
        ),
        "koff_ab": ParamLocalSetting(
            name_setting=NameSetting("koff_ab", "", ("temperature",)),
            value=100.0,
            min=0.0,
            max=1e6,
            vary=True,
        ),
        "kon_ab": ParamLocalSetting(
            name_setting=NameSetting("kon_ab", "", ("temperature",)),
            expr="{koff_ab} / max({kd_ab}, 1e-32)",
        ),

        # Binding: apo A + ligand <-> C
        "kd_ac": ParamLocalSetting(
            name_setting=NameSetting("kd_ac", "", ("temperature",)),
            value=1e-6,
            min=0.0,
            max=1.0,
            vary=True,
        ),
        "koff_ac": ParamLocalSetting(
            name_setting=NameSetting("koff_ac", "", ("temperature",)),
            value=100.0,
            min=0.0,
            max=1e6,
            vary=True,
        ),
        "kon_ac": ParamLocalSetting(
            name_setting=NameSetting("kon_ac", "", ("temperature",)),
            expr="{koff_ac} / max({kd_ac}, 1e-32)",
        ),

        # Bound-state conversion: B <-> D
        "kex_bd": ParamLocalSetting(
            name_setting=NameSetting("kex_bd", "", ("temperature",)),
            value=1000.0,
            min=0.0,
            max=1e6,
            vary=True,
        ),
        "keq_bd": ParamLocalSetting(
            name_setting=NameSetting("keq_bd", "", ("temperature",)),
            value=1.0,
            min=0.0,
            max=100.0,
            vary=True,
        ),

        # Bound-state conversion: C <-> D
        # keq_cd is derived to enforce detailed balance around the cycle:
        #
        # A+L <-> B <-> D
        # A+L <-> C <-> D
        #
        # d/b = keq_bd
        # c/aL = 1/kd_ac
        # b/aL = 1/kd_ab
        # therefore d/c = keq_cd = keq_bd * kd_ac / kd_ab
        "keq_cd": ParamLocalSetting(
            name_setting=NameSetting("keq_cd", "", ("temperature",)),
            expr="{keq_bd} * {kd_ac} / max({kd_ab}, 1e-32)",
        ),
        "kex_cd": ParamLocalSetting(
            name_setting=NameSetting("kex_cd", "", ("temperature",)),
            value=1000.0,
            min=0.0,
            max=1e6,
            vary=True,
        ),

        # Concentrations
        "c_a": ParamLocalSetting(
            name_setting=NameSetting("c_a", "", TPL),
            expr=f"calc_conc({p_total},{l_total},{{kd_ab}},{{kd_ac}},{{keq_bd}})['a']",
        ),
        "c_l": ParamLocalSetting(
            name_setting=NameSetting("c_l", "", TPL),
            expr=f"calc_conc({p_total},{l_total},{{kd_ab}},{{kd_ac}},{{keq_bd}})['l']",
        ),
        "c_b": ParamLocalSetting(
            name_setting=NameSetting("c_b", "", TPL),
            expr=f"calc_conc({p_total},{l_total},{{kd_ab}},{{kd_ac}},{{keq_bd}})['b']",
        ),
        "c_c": ParamLocalSetting(
            name_setting=NameSetting("c_c", "", TPL),
            expr=f"calc_conc({p_total},{l_total},{{kd_ab}},{{kd_ac}},{{keq_bd}})['c']",
        ),
        "c_d": ParamLocalSetting(
            name_setting=NameSetting("c_d", "", TPL),
            expr=f"calc_conc({p_total},{l_total},{{kd_ab}},{{kd_ac}},{{keq_bd}})['d']",
        ),

        # ChemEx rates
        "kab": ParamLocalSetting(
            name_setting=NameSetting("kab", "", TPL),
            expr="{kon_ab} * {c_l}",
        ),
        "kba": ParamLocalSetting(
            name_setting=NameSetting("kba", "", TPL),
            expr="{koff_ab}",
        ),
        "kac": ParamLocalSetting(
            name_setting=NameSetting("kac", "", TPL),
            expr="{kon_ac} * {c_l}",
        ),
        "kca": ParamLocalSetting(
            name_setting=NameSetting("kca", "", TPL),
            expr="{koff_ac}",
        ),

        # B <-> D
        "kbd": ParamLocalSetting(
            name_setting=NameSetting("kbd", "", TPL),
            expr="{keq_bd} * {kex_bd} / (1.0 + {keq_bd})",
        ),
        "kdb": ParamLocalSetting(
            name_setting=NameSetting("kdb", "", TPL),
            expr="{kex_bd} / (1.0 + {keq_bd})",
        ),

        # C <-> D
        "kcd": ParamLocalSetting(
            name_setting=NameSetting("kcd", "", TPL),
            expr="{keq_cd} * {kex_cd} / (1.0 + {keq_cd})",
        ),
        "kdc": ParamLocalSetting(
            name_setting=NameSetting("kdc", "", TPL),
            expr="{kex_cd} / (1.0 + {keq_cd})",
        ),

        # Populations
        "pa": ParamLocalSetting(
            name_setting=NameSetting("pa", "", TPL),
            expr=f"{{c_a}} / {p_total}",
        ),
        "pb": ParamLocalSetting(
            name_setting=NameSetting("pb", "", TPL),
            expr=f"{{c_b}} / {p_total}",
        ),
        "pc": ParamLocalSetting(
            name_setting=NameSetting("pc", "", TPL),
            expr=f"{{c_c}} / {p_total}",
        ),
        "pd": ParamLocalSetting(
            name_setting=NameSetting("pd", "", TPL),
            expr=f"{{c_d}} / {p_total}",
        ),
    }


def register() -> None:
    model_factory.register(
        name=NAME,
        setting_maker=make_settings_4st_binding_cyclic_cs_if,
    )

    user_functions = {
        "calc_conc": calculate_concentrations,
    }

    user_function_registry.register(name=NAME, user_functions=user_functions)