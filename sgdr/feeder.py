"""The physical layer of M2: a SimBench rural MV feeder wrapped so the
Stackelberg game can be audited with AC power flow.

Mapping (see notebook 03 §2 and LEDGER M2-A*):
- every SimBench load = one game household (N = 96); its budget C_n is
  proportional to its nameplate p_mw (equal wealth per kW of appliance);
- the K utility companies all inject at the HV/MV substation — the game
  decides HOW MUCH each household consumes, the feeder then says whether
  that consumption is physically acceptable;
- DER (sgen) is switched off: M2 is a load-only story (LEDGER M2-A3);
- reactive power follows each load's nameplate Q/P ratio.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandapower as pp
import pandapower.topology as top
import networkx as nx

FEEDER_CODE = "1-MV-rural--0-sw"
VMIN_PU = 0.965          # SimBench MV planning band (net.bus.min_vm_pu)
LOADING_MAX = 100.0      # % thermal limit for lines and trafos
VTOL_PU = 2e-4           # feasibility tolerance: 0.02% voltage — far below
ITOL_PCT = 0.1           # any real measurement; avoids exactness artifacts
                         # in the dual ascent (LEDGER M2-A7)


@dataclass
class Feeder:
    net: "pp.pandapowerNet"
    load_idx: np.ndarray          # pandapower load indices = game households
    load_bus: np.ndarray          # bus of each household
    p_nameplate: np.ndarray       # nameplate MW per household
    qratio: np.ndarray            # nameplate Q/P per household
    dist_ohm: np.ndarray          # electrical distance |Z| substation->bus [ohm]

    @property
    def N(self) -> int:
        return len(self.load_idx)

    @property
    def p_total(self) -> float:
        return float(self.p_nameplate.sum())


def load_feeder() -> Feeder:
    import simbench as sb

    net = sb.get_simbench_net(FEEDER_CODE)
    net.sgen.in_service = False              # DER off (M2 scope)
    net.load["scaling"] = 1.0

    load_idx = net.load.index.to_numpy()
    load_bus = net.load.bus.to_numpy()
    p_np = net.load.p_mw.to_numpy().copy()
    qratio = (net.load.q_mvar / net.load.p_mw).to_numpy()

    # electrical distance: shortest path over |Z| of lines + trafos
    g = nx.Graph()
    for _, ln in net.line.iterrows():
        z = ln.length_km * np.hypot(ln.r_ohm_per_km, ln.x_ohm_per_km)
        g.add_edge(int(ln.from_bus), int(ln.to_bus), weight=float(z))
    for _, tr in net.trafo.iterrows():
        z_ohm = (tr.vk_percent / 100.0) * (tr.vn_lv_kv ** 2) / tr.sn_mva
        g.add_edge(int(tr.hv_bus), int(tr.lv_bus), weight=float(z_ohm))
    src = int(net.ext_grid.bus.iloc[0])
    dist = nx.single_source_dijkstra_path_length(g, src)
    dist_ohm = np.array([dist[int(b)] for b in load_bus])

    return Feeder(net=net, load_idx=load_idx, load_bus=load_bus,
                  p_nameplate=p_np, qratio=qratio, dist_ohm=dist_ohm)


def audit(fdr: Feeder, p_mw: np.ndarray) -> dict:
    """Set household consumptions p_mw [N], run AC PF, grade the state.

    Returns the physics verdict the copperplate game never sees.
    """
    net = fdr.net
    net.load.loc[fdr.load_idx, "p_mw"] = p_mw
    net.load.loc[fdr.load_idx, "q_mvar"] = p_mw * fdr.qratio
    try:
        pp.runpp(net, calculate_voltage_angles=True, init="dc")
    except pp.LoadflowNotConverged:
        return {"converged": False}

    mv = net.bus.vn_kv < 100.0
    vm = net.res_bus.vm_pu[mv]
    line_load = net.res_line.loading_percent.max()
    trafo_load = net.res_trafo.loading_percent.max()
    return {
        "converged": True,
        "min_vm": float(vm.min()),
        "argmin_bus": int(vm.idxmin()),
        "max_line_loading": float(line_load),
        "max_trafo_loading": float(trafo_load),
        "losses_mw": float(net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum()),
        "vm_load_bus": net.res_bus.vm_pu.loc[fdr.load_bus].to_numpy(),
        "feasible": bool(vm.min() >= VMIN_PU - VTOL_PU
                         and line_load <= LOADING_MAX + ITOL_PCT
                         and trafo_load <= LOADING_MAX + ITOL_PCT),
        "v_violation": float(max(0.0, VMIN_PU - vm.min())),
        "i_violation": float(max(0.0, max(line_load, trafo_load) - LOADING_MAX)),
    }


def sensitivities(fdr: Feeder, p_mw: np.ndarray, dp: float = 0.05) -> dict:
    """Numerical constraint sensitivities at operating point p_mw.

    S_v[b, n] = -d(vm at load-bus b)/d(p_n)  [pu/MW]  — per-bus voltage
    sensitivities (the dual ascent prices each violated BUS separately;
    a single global-min constraint whack-a-moles between feeder branches
    — see LEDGER M2-D1).
    s_i[n]    =  d(max loading)/d(p_n)  [%/MW]  — thermal kept global.

    One PF per household; per-bus columns come free from the same runs.
    """
    base = audit(fdr, p_mw)
    assert base["converged"]
    S_v = np.zeros((fdr.N, fdr.N))       # [monitored load-bus b, household n]
    s_i = np.zeros(fdr.N)
    base_vm = base["vm_load_bus"]
    base_load = max(base["max_line_loading"], base["max_trafo_loading"])
    for n in range(fdr.N):
        p2 = p_mw.copy()
        p2[n] += dp
        r = audit(fdr, p2)
        if not r["converged"]:
            continue
        S_v[:, n] = -(r["vm_load_bus"] - base_vm) / dp
        s_i[n] = (max(r["max_line_loading"], r["max_trafo_loading"])
                  - base_load) / dp
    return {"S_v": np.clip(S_v, 0.0, None), "s_i": np.clip(s_i, 0.0, None),
            "base": base}
