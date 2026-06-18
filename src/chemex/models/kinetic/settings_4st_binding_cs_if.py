from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import root

from chemex.configuration.conditions import Conditions
from chemex.models.factory import model_factory
from chemex.parameters.setting import NameSetting, ParamLocalSetting
from chemex.parameters.userfunctions import user_function_registry
from chemex.typing import Array

NAME = "4st_binding_cs_if"

TPL = ("temperature", "p_total", "l_total")

# Combined Conformational Selection + Induced Fit (CS+IF) 4-state binding model.
#
# Exchange topology (square):
#
#   A (free, ground) <-kge/keg-> B (free, excited)      [CS: apo conformational exchange]
#        |  kac/kca                    |  kbd/kdb
#   A+L <-> C (bound, ground) <-kcd/kdc-> D (bound, excited)  [IF: bound conformational exchange]
#
# States:
#   A: apo ground state  (major free-protein state)
#   B: apo excited state (minor free-protein state, CS pre-existing conformation)
#   C: ligand-bound ground state (A+L binding pathway)
#   D: ligand-bound excited state (B+L binding OR C->D induced fit)
#
# Thermodynamic equilibrium constants:
#   K_AB = [B]/[A] = k_apo             (apo CS equilibrium)
#   Kd_g = [A][L]/[C] = kd_g          (ground-state dissociation constant)
#   Kd_b = [B][L]/[D] = kd_b          (excited-state dissociation constant)
#   K_CD = [D]/[C] = k_apo*kd_g/kd_b  (bound IF equilibrium, from detailed balance)
#
# Detailed balance constraints:
#   keg = kge / k_apo             (B->A rate from K_AB detailed balance)
#   kbg = kgb * kd_b / (k_apo * kd_g)  (D->C rate from K_CD detailed balance)
#
# Independent kinetic parameters (7):
#   k_apo  : equilibrium constant [B]/[A] for free protein (CS indicator; pb/pa at zero [L])
#   kge    : A->B rate constant (s^-1)
#   kd_g   : dissociation constant for ground-state binding A+L<->C (M)
#   kd_b   : dissociation constant for excited-state binding B+L<->D (M)
#   koff_g : off-rate for C->A+L (s^-1)
#   koff_b : off-rate for D->B+L (s^-1)
#   kgb    : C->D rate constant (IF forward rate, s^-1)
#
# Physical limits:
#   Pure CS  : kd_b << kd_g (excited state has higher affinity), kgb -> 0
#   Pure IF  : kd_b >> kd_g (only ground state binds, then C<->D IF occurs)
#   CS + IF  : both pathways active, kd_b < kd_g AND kgb significant
#
# Reference: combined CS+IF model as in Walinda et al. PNAS 2024, doi:10.1073/pnas.2317747121


def calculate_residuals(
    vars: Array,
    p_total: float,
    l_total: float,
    k_apo: float,
    kd_g: float,
    kd_b: float,
) -> Array:
    pa, pb, pc, pd, l_free = vars
    return np.array(
        [
            pa + pb + pc + pd - 1.0,
            pb - k_apo * pa,
            pc - (pa * l_free / max(kd_g, 1e-10)),
            pd - (pb * l_free / max(kd_b, 1e-10)),
            l_free + (pc + pd) * p_total - l_total,
        ],
    )


@lru_cache(maxsize=100)
def calculate_concentrations(
    p_total: float,
    l_total: float,
    k_apo: float,
    kd_g: float,
    kd_b: float,
) -> dict[str, float]:
    vars_start = (0.9, 0.05, 0.025, 0.025, l_total)
    results = root(
        calculate_residuals,
        vars_start,
        args=(p_total, l_total, k_apo, kd_g, kd_b),
    )
    pa, pb, pc, pd, l_free = results["x"]
    return {"pa": pa, "pb": pb, "pc": pc, "pd": pd, "l_free": l_free}


def make_settings_4st_binding_cs_if(
    conditions: Conditions,
) -> dict[str, ParamLocalSetting]:
    p_total = conditions.p_total
    l_total = conditions.l_total
    if p_total is None or l_total is None:
        msg = (
            f"'p_total' and 'l_total' must be specified to use the '{NAME}' model"
        )
        raise ValueError(msg)

    calc = f"calc_conc({p_total}, {l_total}, {{k_apo}}, {{kd_g}}, {{kd_b}})"

    return {
        # --- CS equilibrium: [B]/[A] in the free protein ---
        "k_apo": ParamLocalSetting(
            name_setting=NameSetting("k_apo", "", ("temperature",)),
            value=0.05,
            min=1e-6,
            vary=True,
        ),

        # --- Apo conformational exchange rates (A <-> B) ---
        # keg = kge / k_apo enforces: kge * pa = keg * pb at equilibrium
        "kge": ParamLocalSetting(
            name_setting=NameSetting("kge", "", ("temperature",)),
            value=100.0,
            min=0.0,
            vary=True,
        ),
        "keg": ParamLocalSetting(
            name_setting=NameSetting("keg", "", ("temperature",)),
            expr="{kge} / max({k_apo}, 1e-100)",
        ),

        # --- Bound conformational exchange rates (C <-> D, IF pathway) ---
        # kbg = kgb * kd_b / (k_apo * kd_g) from detailed balance K_AB * K_CD = Kd_g / Kd_b
        "kgb": ParamLocalSetting(
            name_setting=NameSetting("kgb", "", ("temperature",)),
            value=500.0,
            min=0.0,
            vary=True,
        ),
        "kbg": ParamLocalSetting(
            name_setting=NameSetting("kbg", "", ("temperature",)),
            expr="{kgb} * {kd_b} / (max({k_apo}, 1e-100) * max({kd_g}, 1e-100))",
        ),

        # --- Dissociation constants ---
        "kd_g": ParamLocalSetting(
            name_setting=NameSetting("kd_g", "", ("temperature",)),
            value=1e-4,
            min=0.0,
            vary=True,
        ),
        "kd_b": ParamLocalSetting(
            name_setting=NameSetting("kd_b", "", ("temperature",)),
            value=1e-5,
            min=0.0,
            vary=True,
        ),

        # --- Binding off-rates ---
        "koff_g": ParamLocalSetting(
            name_setting=NameSetting("koff_g", "", ("temperature",)),
            value=100.0,
            min=0.0,
            vary=True,
        ),
        "kon_g": ParamLocalSetting(
            name_setting=NameSetting("kon_g", "", ("temperature",)),
            expr="{koff_g} / max({kd_g}, 1e-100)",
        ),
        "koff_b": ParamLocalSetting(
            name_setting=NameSetting("koff_b", "", ("temperature",)),
            value=100.0,
            min=0.0,
            vary=True,
        ),
        "kon_b": ParamLocalSetting(
            name_setting=NameSetting("kon_b", "", ("temperature",)),
            expr="{koff_b} / max({kd_b}, 1e-100)",
        ),

        # --- Equilibrium populations and free ligand (concentration-dependent) ---
        "l_free": ParamLocalSetting(
            name_setting=NameSetting("l_free", "", TPL),
            expr=f"{calc}['l_free']",
        ),
        "pa": ParamLocalSetting(
            name_setting=NameSetting("pa", "", TPL),
            expr=f"{calc}['pa']",
        ),
        "pb": ParamLocalSetting(
            name_setting=NameSetting("pb", "", TPL),
            expr=f"{calc}['pb']",
        ),
        "pc": ParamLocalSetting(
            name_setting=NameSetting("pc", "", TPL),
            expr=f"{calc}['pc']",
        ),
        "pd": ParamLocalSetting(
            name_setting=NameSetting("pd", "", TPL),
            expr=f"{calc}['pd']",
        ),

        # --- ChemEx transition rate labels (kXY = rate from state X to state Y) ---
        "kab": ParamLocalSetting(
            name_setting=NameSetting("kab", "", TPL),
            expr="{kge}",
        ),
        "kba": ParamLocalSetting(
            name_setting=NameSetting("kba", "", TPL),
            expr="{keg}",
        ),
        "kac": ParamLocalSetting(
            name_setting=NameSetting("kac", "", TPL),
            expr="{kon_g} * {l_free}",
        ),
        "kca": ParamLocalSetting(
            name_setting=NameSetting("kca", "", TPL),
            expr="{koff_g}",
        ),
        "kbd": ParamLocalSetting(
            name_setting=NameSetting("kbd", "", TPL),
            expr="{kon_b} * {l_free}",
        ),
        "kdb": ParamLocalSetting(
            name_setting=NameSetting("kdb", "", TPL),
            expr="{koff_b}",
        ),
        "kcd": ParamLocalSetting(
            name_setting=NameSetting("kcd", "", TPL),
            expr="{kgb}",
        ),
        "kdc": ParamLocalSetting(
            name_setting=NameSetting("kdc", "", TPL),
            expr="{kbg}",
        ),
    }


def register() -> None:
    model_factory.register(name=NAME, setting_maker=make_settings_4st_binding_cs_if)
    user_functions = {"calc_conc": calculate_concentrations}
    user_function_registry.register(name=NAME, user_functions=user_functions)
