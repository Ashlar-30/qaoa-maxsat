#!/usr/bin/env python3
# Copyright 2026 Ashwin Kumar Baskaran
# SPDX-License-Identifier: MIT
"""
QAOA solver for Weighted Partial MAX-3-SAT, using the CJ2 encoding.

Reads a DIMACS .cnf or .wcnf file, turns it into a QUBO, runs QAOA on
PennyLane (lightning.qubit on CPU or lightning.gpu on an NVIDIA card), and
checks the result against RC2 if python-sat is installed.

    python src/qaoa_wcnf_mod.py instances/Fifteen.cnf --p 1 --shots 65536 --seed 42

--help lists every option. Output files all start with wcnf_ and go into
whatever folder you run it from.

References:
    Farhi, Goldstone, Gutmann (2014), arXiv:1411.4028 (QAOA)
    Ansotegui, Levy (2024), arXiv:2403.00182 (CJ2 encoding)
    Rodriguez-Farres, Ballester, Ansotegui, Levy, Cerquides (2024), ICCS 2024,
        doi:10.1007/978-3-031-63778-0_20 (3-SAT gadgets for quantum annealers)
"""

import sys, os, csv, time, argparse, math, warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Tuple, Dict

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize

try:
    from pysat.formula import WCNF as PYSAT_WCNF
    from pysat.examples.rc2 import RC2 as PYSAT_RC2
    HAS_PYSAT = True
except ImportError:
    HAS_PYSAT = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore",
    message=r".*disp.*iprint.*L-BFGS-B.*", category=DeprecationWarning)


# Exceptions and small file helpers

class StagnationError(Exception):
    """Triggers early exit from L-BFGS-B when no improvement for N evals."""


def _safe_save_fig(fig, filepath: str, **kwargs) -> None:
    try:
        fig.savefig(filepath, **kwargs)
    except Exception as e:
        print(f"  [IO WARN] Could not save {filepath}: {e}")


def _safe_write_text(filepath: str, header: str, body: str) -> None:
    try:
        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            if header:
                f.write(header + "\n\n")
            f.write(body + "\n")
    except Exception as e:
        print(f"  [IO WARN] Could not write {filepath}: {e}")


# Run settings

@dataclass
class QAOAConfig:
    p: int = 1
    seed: int = 42
    n_shots: int = 65536
    backend: str = "cpu"
    n_restarts: Optional[int] = None
    draw: bool = False
    plot_samples: bool = False
    plot_landscape: bool = False
    landscape_grid: int = 60
    landscape_zoom: float = 0.0
    landscape_fine: bool = False
    verbose: bool = False


# Reading DIMACS .cnf / .wcnf files

class WeightedCNFProblem:
    """
    Parses DIMACS .cnf / .wcnf into:
      clauses : list of [(var_0based, sign), ...] triplets (strict 3-SAT)
      weights : float per clause (1.0 for plain CNF, w_j for WCNF)
    """

    def __init__(self, n_vars: int, clauses: List[List[Tuple[int, int]]],
                 weights: List[float], source: str = "unknown",
                 is_weighted: bool = False):
        self.n_vars = n_vars
        self.clauses = clauses
        self.weights = weights
        self.n_clauses = len(clauses)
        self.source = source
        self.is_weighted = is_weighted

    @classmethod
    def from_file(cls, filename: str) -> "WeightedCNFProblem":
        clauses, weights, n_vars = [], [], 0
        is_weighted = False

        with open(filename, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line[0] in ("c", "%"):
                    continue
                if line[0] == "p":
                    parts = line.split()
                    is_weighted = (parts[1].lower() == "wcnf")
                    n_vars = int(parts[2])
                    continue
                tokens = line.split()
                if "c" in tokens:
                    tokens = tokens[:tokens.index("c")]
                if not tokens:
                    continue
                if is_weighted:
                    w = float(tokens[0])
                    lits = [int(x) for x in tokens[1:] if x != "0"]
                else:
                    w = 1.0
                    lits = [int(x) for x in tokens if x != "0"]
                if not lits:
                    continue
                if len(lits) != 3 or len(set(abs(l) for l in lits)) != 3:
                    raise ValueError(
                        f"Syntax error in {filename}: clause ({lits}) must have "
                        f"exactly 3 unique variables. CJ2 strictly requires 3-SAT."
                    )
                clauses.append([(abs(l) - 1, 1 if l > 0 else -1) for l in lits])
                weights.append(w)

        return cls(n_vars, clauses, weights,
                   source=os.path.basename(filename),
                   is_weighted=is_weighted)

    @property
    def n_qubits(self) -> int:
        # CJ2: one ancilla qubit per clause
        return self.n_vars + self.n_clauses

    @property
    def ratio(self) -> float:
        return self.n_clauses / self.n_vars if self.n_vars > 0 else 0.0

    @property
    def total_weight(self) -> float:
        return sum(self.weights)

    def weight_summary(self) -> str:
        c = Counter(self.weights)
        return " | ".join(f"w={w:.0f}x{cnt}" for w, cnt in sorted(c.items()))

    def __repr__(self) -> str:
        return (f"WeightedCNFProblem(n={self.n_vars}, m={self.n_clauses}, "
                f"weighted={self.is_weighted}, source='{self.source}')")


# CJ2 encoding: one extra variable per clause, scaled by the clause weight

class WeightedCJ2QUBOBuilder:
    """
    For each clause j with literals (v1,s1),(v2,s2),(v3,s3) and weight w_j,
    the upper-triangular QUBO is incremented by:

        Q[v1,v1] += w_j * (s1 - s1*s3)
        Q[v2,v2] += w_j * (-2*s2)
        Q[v3,v3] += w_j * (s3 - s1*s3)
        Q[bj,bj] += w_j * -(1 - s1 + s2*s3)
        Q[v1,v3] += w_j * 2*s1*s3
        Q[v1,bj] += w_j * -2*s1
        Q[v2,bj] += w_j * 2*s2
        Q[v3,bj] += w_j * -2*s3

    where bj = n_vars + j is the ancilla qubit for clause j.
    Higher clause weight => deeper energy penalty for violation.
    """

    def __init__(self, problem: WeightedCNFProblem):
        self.problem = problem
        self._Q: Optional[np.ndarray] = None

    def build(self) -> np.ndarray:
        if self._Q is not None:
            return self._Q

        n_q = self.problem.n_qubits
        Q = np.zeros((n_q, n_q), dtype=np.float64)

        def acc(i: int, j: int, val: float) -> None:
            if i > j:
                i, j = j, i
            Q[i, j] += val

        for j, clause in enumerate(self.problem.clauses):
            (v1, s1), (v2, s2), (v3, s3) = clause
            bj = self.problem.n_vars + j
            w = self.problem.weights[j]

            acc(v1, v1, w * float(s1 - s1 * s3))
            acc(v2, v2, w * float(-2 * s2))
            acc(v3, v3, w * float(s3 - s1 * s3))
            acc(bj, bj, w * float(-(1 - s1 + s2 * s3)))
            acc(v1, v3, w * float(2 * s1 * s3))
            acc(v1, bj, w * float(-2 * s1))
            acc(v2, bj, w * float(2 * s2))
            acc(v3, bj, w * float(-2 * s3))

        self._Q = Q
        return Q

    @property
    def Q(self) -> np.ndarray:
        return self.build()

    def print_matrix(self, verbose: bool = False) -> None:
        Q = self.Q
        n, m = self.problem.n_vars, self.problem.n_clauses
        n_q = n + m
        labels = [f"x{i+1}" for i in range(n)] + [f"b{i+1}" for i in range(m)]
        cw = 8
        header = "  Weighted QUBO matrix:\n"
        header += " " * cw + "".join(f"{l:>{cw}}" for l in labels) + "\n"
        header += " " * cw + "-" * (cw * n_q) + "\n"
        body = ""
        for i in range(n_q):
            row = f"{labels[i]:>{cw}}"
            for j in range(n_q):
                if j < i:
                    row += ".".rjust(cw)
                else:
                    v = Q[i, j]
                    if abs(v) < 1e-12:
                        row += ".".rjust(cw)
                    elif v == int(v):
                        row += f"{int(v):>{cw}}"
                    else:
                        row += f"{v:>{cw}.2f}"
            body += row + "\n"
        _safe_write_text("wcnf_qubo_matrix.txt", header.strip(), body)
        if verbose or n_q <= 12:
            print("\n" + header + body)
        else:
            print(f"  [QUBO] Matrix ({n_q}x{n_q}) saved to wcnf_qubo_matrix.txt")


# QUBO to Ising (h, J), using x = (1 + z) / 2

class IsingHamiltonian:
    """
    Converts QUBO Q[i,j] over x in {0,1} to Ising (h, J) over z in {-1,+1}.
    Diagonals -> h linear fields + constant offset.
    Off-diagonals -> J couplings + corrections to h[i], h[j].
    The PennyLane Hamiltonian = sum h_i Z_i + sum J_ij Z_i Z_j is the QAOA
    cost operator (only Z and ZZ terms, native to any 2-body hardware).
    """

    def __init__(self, Q: np.ndarray, n_qubits: int):
        self.n_qubits = n_qubits
        self.h, self.J, self.offset = self._qubo_to_ising(Q)
        self.z_terms = [(i, self.h[i]) for i in range(n_qubits) if abs(self.h[i]) > 1e-12]
        self.zz_terms = [(i, j, w) for i, j, w in self.J if abs(w) > 1e-12]
        self.hamiltonian = self._build_hamiltonian()

    def _qubo_to_ising(self, Q: np.ndarray):
        h = np.zeros(self.n_qubits)
        J: List[Tuple] = []
        offset = 0.0
        for i in range(self.n_qubits):
            qii = Q[i, i]
            if abs(qii) > 1e-12:
                offset += 0.5 * qii
                h[i] -= 0.5 * qii
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                qij = Q[i, j]
                if abs(qij) > 1e-12:
                    offset += 0.25 * qij
                    h[i] -= 0.25 * qij
                    h[j] -= 0.25 * qij
                    J.append((i, j, 0.25 * qij))
        return h, J, offset

    def _build_hamiltonian(self):
        coeffs, ops = [], []
        for i in range(self.n_qubits):
            if abs(self.h[i]) > 1e-12:
                coeffs.append(self.h[i])
                ops.append(qml.PauliZ(i))
        for i, j, w in self.J:
            if abs(w) > 1e-12:
                coeffs.append(w)
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
        if not coeffs:
            coeffs, ops = [0.0], [qml.Identity(wires=0)]
        return qml.dot(coeffs, ops)


# PennyLane devices: an exact one for optimising, a shot-based one for sampling

class QAOADevices:
    """
    Two devices to avoid PennyLane mode conflicts between adjoint
    differentiation (statevector) and shot-based bitstring sampling.
    """

    def __init__(self, n_qubits: int, backend: str):
        self.n_qubits = n_qubits
        self.backend = backend
        self.analytic_dev, self.sampling_dev = self._make()

    def _make(self):
        if self.backend == "gpu":
            try:
                import pennylane_lightning  # noqa: F401
                a = qml.device("lightning.gpu", wires=self.n_qubits)
                s = qml.device("lightning.gpu", wires=self.n_qubits)
                print(f"      Using: lightning.gpu x2 "
                      f"(Separate Analytic+Sampling) (wires={self.n_qubits})")
                return a, s
            except Exception:
                print("      lightning.gpu unavailable -- falling back to lightning.qubit")
        a = qml.device("lightning.qubit", wires=self.n_qubits)
        s = qml.device("lightning.qubit", wires=self.n_qubits)
        print(f"      Using: lightning.qubit (wires={self.n_qubits})")
        return a, s


# The circuit: Hadamards, then p rounds of cost and mixer, then measure

class QAOACircuit:
    """
    Standard QAOA: H^{otimes n} -> [exp(-i gamma_l H_C) * exp(-i beta_l H_M)]^p.
    Adjoint differentiation for analytic gradients via qml.grad.
    Raw (gamma, beta) angles -- no Fourier reparameterisation.
    """

    def __init__(self, devices: QAOADevices, ising: IsingHamiltonian,
                 p: int, n_shots: int, n_vars: int):
        self.devices = devices
        self.ising = ising
        self.p = p
        self.n_qubits = ising.n_qubits
        self.n_shots = n_shots
        self.n_vars = n_vars
        self._build()

    def _apply_layers(self, params, decompose_zz: bool = False):
        gamma, beta = params[:self.p], params[self.p:]
        for w in range(self.n_qubits):
            qml.Hadamard(wires=w)
        for layer in range(self.p):
            g, b = gamma[layer], beta[layer]
            for qi, wt in self.ising.z_terms:
                qml.RZ(2.0 * g * wt, wires=qi)
            for qi, qj, wt in self.ising.zz_terms:
                if decompose_zz:
                    # IsingZZ rewritten as CNOT-RZ-CNOT for hardware-native gate counting
                    phi = 2.0 * g * wt
                    qml.CNOT(wires=[qi, qj])
                    qml.RZ(phi, wires=qj)
                    qml.CNOT(wires=[qi, qj])
                else:
                    qml.IsingZZ(2.0 * g * wt, wires=[qi, qj])
            for w in range(self.n_qubits):
                qml.RX(2.0 * b, wires=w)

    def _build(self):
        H = self.ising.hamiltonian
        adev = self.devices.analytic_dev
        sdev = self.devices.sampling_dev

        @qml.qnode(adev, diff_method="adjoint", interface="autograd")
        def _cgrad(params):
            self._apply_layers(params)
            return qml.expval(H)

        @qml.qnode(adev, diff_method=None, interface="autograd")
        def _ceval(params):
            self._apply_layers(params)
            return qml.expval(H)

        @qml.qnode(sdev, diff_method=None, shots=self.n_shots)
        def _csample(params):
            self._apply_layers(params)
            return qml.counts(wires=list(range(self.n_qubits)))

        _gfn = qml.grad(_cgrad)

        def energy(params: np.ndarray) -> float:
            return float(_ceval(pnp.array(params, requires_grad=False)))

        def energy_and_grad(params: np.ndarray) -> Tuple[float, np.ndarray]:
            pa = pnp.array(params, requires_grad=True)
            grad = _gfn(pa)
            e = float(_ceval(pnp.array(params, requires_grad=False)))
            return e, np.array(grad, dtype=np.float64)

        self.energy = energy
        self.energy_and_grad = energy_and_grad
        self._csample = _csample

    def sample(self, params: np.ndarray) -> Dict:
        return self._csample(params)

    def gate_counts(self, decompose_zz: bool = True):
        dp = np.array([1.93] * self.p + [0.34] * self.p)

        @qml.qnode(self.devices.analytic_dev, diff_method=None)
        def cdraw(params):
            self._apply_layers(params, decompose_zz=decompose_zz)
            return qml.expval(qml.PauliZ(0))

        specs = qml.specs(cdraw)(dp)["resources"]
        return {
            "wires": self.n_qubits,
            "depth": specs.depth,
            "num_gates": specs.num_gates,
            "gate_types": dict(specs.gate_types),
        }

    def print_gate_counts(self) -> None:
        info = self.gate_counts(decompose_zz=True)
        gt = info["gate_types"]
        n_h = gt.get("Hadamard", 0)
        n_rz = gt.get("RZ", 0)
        n_rx = gt.get("RX", 0)
        n_cnot = gt.get("CNOT", 0)
        total = n_h + n_rz + n_rx + n_cnot

        print(f"\n  [Circuit] {self.n_qubits} qubits  |  p={self.p}  |  depth = {info['depth']}")
        print(f"  -----------------------------------")
        print(f"    Hadamard  : {n_h:>6}")
        print(f"    RZ        : {n_rz:>6}")
        print(f"    RX        : {n_rx:>6}")
        print(f"    CNOT      : {n_cnot:>6}   ({n_cnot // 2} ZZ couplings)")
        print(f"  -----------------------------------")
        print(f"    TOTAL     : {total:>6}\n")

    def draw_and_save(self, label: str) -> None:
        dp = np.array([1.93] * self.p + [0.34] * self.p)

        @qml.qnode(self.devices.analytic_dev, diff_method=None)
        def cdraw(params):
            self._apply_layers(params, decompose_zz=True)
            return qml.expval(qml.PauliZ(0))

        specs = qml.specs(cdraw)(dp)["resources"]
        print(f"  [Circuit] Wires={self.n_qubits}  "
              f"Gates={specs.num_gates}  Depth={specs.depth}")
        for gt, cnt in sorted(specs.gate_types.items(), key=lambda x: -x[1]):
            print(f"    {gt:<20}: {cnt}")

        txt = qml.draw(cdraw, max_length=240)(dp)
        _safe_write_text(
            f"wcnf_circuit_p{self.p}.txt",
            f"Weighted QAOA Circuit | {label} | p={self.p} | {self.n_qubits} qubits",
            txt)

        fw = max(12.0, (specs.num_gates / max(self.n_qubits, 1)) * 0.55 + self.n_qubits * 0.2)
        fh = max(4.0, self.n_qubits * 0.45)
        qml.drawer.use_style("pennylane")
        fig, ax = qml.draw_mpl(cdraw, fig_kwargs={"figsize": (fw, fh)})(dp)
        ax.set_title(f"Weighted QAOA Circuit | {label} | p={self.p}", fontsize=9, pad=8)
        _safe_save_fig(fig, f"wcnf_circuit_p{self.p}.png", dpi=300,
                       bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)
        print(f"  [Circuit PNG] Saved wcnf_circuit_p{self.p}.png\n")


# Bookkeeping for one L-BFGS-B restart

class RestartState:
    def __init__(self, restart_id: int):
        self.restart_id = restart_id
        self.eval_count = 0
        self.best_energy = float("inf")
        self.best_angles: Optional[np.ndarray] = None
        self.last_improvement_eval = 0
        self.eval_log: List[Dict] = []

    def update(self, params: np.ndarray, energy: float) -> bool:
        self.eval_count += 1
        improved = energy < self.best_energy
        if improved:
            self.best_energy = energy
            self.best_angles = params.copy()
            self.last_improvement_eval = self.eval_count
            self.eval_log.append({
                "restart": self.restart_id,
                "eval": self.eval_count,
                "energy": energy,
            })
        return improved


# Parameter search: SAT-Init start, then perturbation, then Latin Hypercube restarts

class WarmStartOptimizer:
    """
    Three-phase QAOA outer-loop optimization:
      Phase 1 -- SAT-Init grid scan around theoretically-optimal angles.
                 For alpha in [3.0, 5.5] (random 3-SAT regime): g* = 1.937/p
                 (Streif & Leib 2020). Otherwise pi/(2p) fallback.
      Phase 2 -- Gaussian perturbations of the Phase-1 best (sigma=0.15, 0.30).
      Phase 3 -- Latin Hypercube space-filling restarts over [-2pi, 2pi]^{2p}.
    Each restart runs L-BFGS-B with adjoint-grad jacobian.
    Stagnation guard: raise StagnationError if no improvement for
                      max(50, 40p) evals -> early exit from current restart.
    """

    def __init__(self, circuit: QAOACircuit, config: QAOAConfig,
                 n_vars: int, n_clauses: int):
        self.circuit = circuit
        self.p = config.p
        self.seed_base = config.seed
        self.verbose = config.verbose
        self.ratio = n_clauses / n_vars if n_vars > 0 else 0.0
        n_req = max(8, 6 + 2 * self.p)
        self.n_restarts = (config.n_restarts
                           if config.n_restarts is not None else n_req)
        seq = np.random.SeedSequence(self.seed_base)
        child1, child2 = seq.spawn(2)
        self.perturb_rng = np.random.default_rng(child1)
        self.lhs_rng = np.random.default_rng(child2)

    def _sat_init(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        # Grid scan over (gamma, beta) calibrated to clause/var ratio
        p = self.p
        energy_fn = self.circuit.energy
        g_star = 1.937 / p if 3.0 <= self.ratio <= 5.5 else math.pi / (2.0 * p)
        g_lo, g_hi = max(0.3, g_star * 0.5), min(math.pi, g_star * 1.8)
        g_range = np.concatenate([np.linspace(-g_hi, -g_lo, 15),
                                  np.linspace(g_lo, g_hi, 15)])
        b_range = np.linspace(-math.pi, math.pi, 10)
        best_x, best_e = None, float("inf")
        second_x, second_e = None, float("inf")
        for g in g_range:
            for b in b_range:
                a = np.zeros(2 * p)
                for i in range(p):
                    a[i] = g * (i + 1)
                    a[p + i] = b * (p - i) / p
                e = float(energy_fn(a))
                if e < best_e:
                    second_e = best_e
                    second_x = best_x.copy() if best_x is not None else None
                    best_e, best_x = e, a.copy()
                elif e < second_e:
                    second_e, second_x = e, a.copy()
        return best_x, second_x

    def _build_lhs(self, n_lhs: int) -> np.ndarray:
        dim = 2 * self.p
        mat = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = self.lhs_rng.permutation(n_lhs)
            jitter = self.lhs_rng.uniform(0.0, 1.0, size=n_lhs)
            span = 4.0 * math.pi
            mat[:, d] = -2.0 * math.pi + (perm + jitter) * (span / max(1, n_lhs))
        return mat

    def _run_one(self, r: int, label: str, x0: np.ndarray,
                 all_log: List) -> Tuple[float, np.ndarray, int]:
        state = RestartState(r)
        threshold = max(50, 40 * self.p)

        def obj(params):
            e, grad = self.circuit.energy_and_grad(params)
            state.update(np.array(params), float(e))
            streak = state.eval_count - state.last_improvement_eval
            if state.eval_count > 30 and streak > threshold:
                raise StagnationError()
            return float(e), np.array(grad, dtype=np.float64)

        try:
            minimize(obj, x0, method="L-BFGS-B", jac=True,
                     options={"maxiter": 300 * self.p, "ftol": 1e-12,
                              "gtol": 1e-8, "maxls": 40})
        except StagnationError:
            if self.verbose:
                print(f"    [WARN] r={r} ({label}): stagnation -- early exit")

        best_x = state.best_angles if state.best_angles is not None else x0
        all_log.extend(state.eval_log)
        return state.best_energy, best_x, state.eval_count

    def run(self) -> Tuple[float, Optional[np.ndarray], List]:
        all_log = []
        best_energy = float("inf")
        best_angles: Optional[np.ndarray] = None
        r = 0

        # Phase 1: SAT-Init
        sat_best, sat_2nd = self._sat_init()
        phase1: List[Tuple[str, np.ndarray]] = []
        if sat_best is not None:
            phase1.append(("SAT-init best", sat_best))
        if sat_2nd is not None:
            phase1.append(("SAT-init 2nd", sat_2nd))

        for label, x0 in phase1:
            e, x, n_ev = self._run_one(r, label, x0, all_log)
            if e < best_energy:
                best_energy, best_angles = e, x.copy()
            print(f"    [{r:3d}] {label:<25s}  E={e:+.6f}  evals={n_ev:4d}")
            r += 1

        # Phase 2: Perturbations of Phase-1 best
        if best_angles is not None:
            for sigma in [0.15, 0.30]:
                label = f"Perturb s={sigma:.2f}"
                x0 = (best_angles.copy()
                      + sigma * self.perturb_rng.standard_normal(2 * self.p))
                e, x, n_ev = self._run_one(r, label, x0, all_log)
                if e < best_energy:
                    best_energy, best_angles = e, x.copy()
                print(f"    [{r:3d}] {label:<25s}  E={e:+.6f}  evals={n_ev:4d}")
                r += 1

        # Phase 3: Latin Hypercube space-filling
        n_lhs = max(0, self.n_restarts - r)
        if n_lhs > 0:
            lhs = self._build_lhs(n_lhs)
            for i in range(n_lhs):
                e, x, n_ev = self._run_one(r, f"LHS[{i}]", lhs[i], all_log)
                if e < best_energy:
                    best_energy, best_angles = e, x.copy()
                print(f"    [{r:3d}] LHS[{i:<20d}]  E={e:+.6f}  evals={n_ev:4d}")
                r += 1

        return best_energy, best_angles, all_log


# RC2 from PySAT as the classical reference, weighted and unweighted

class RC2Processor:
    """
    Runs PySAT's RC2 MaxSAT solver twice:
      - Weighted: with the actual clause weights (scaled to integers if needed).
      - Unweighted: with weight=1 per clause to get the satisfied-count optimum.
    Returns (opt_unweighted_count, opt_weighted_sum, time_seconds).
    """

    def __init__(self, problem: WeightedCNFProblem):
        self.problem = problem

    def solve(self) -> Tuple[Optional[int], Optional[float], float]:
        if not HAS_PYSAT:
            return None, None, 0.0
        t0 = time.time()

        # Weighted pass: PySAT needs integer weights, so floats are scaled by 1000.
        wcnf_w = PYSAT_WCNF()
        for clause, w in zip(self.problem.clauses, self.problem.weights):
            lits = [(v + 1) * s for v, s in clause]
            iw = int(round(w * 1000)) if (w != int(w)) else int(w)
            if iw < 1:
                iw = 1
            wcnf_w.append(lits, weight=iw)
        with PYSAT_RC2(wcnf_w) as rc2:
            model = rc2.compute()
            if model is None:
                return None, None, time.time() - t0
            total_w_scaled = sum(
                int(round(w * 1000)) if (w != int(w)) else int(w)
                for w in self.problem.weights
            )
            opt_w_scaled = total_w_scaled - rc2.cost
            scale_back = (1000.0 if any(w != int(w) for w in self.problem.weights)
                          else 1.0)
            opt_weighted = opt_w_scaled / scale_back

        # Unweighted pass: every clause weighs 1, so RC2.cost is the unsatisfied count.
        wcnf_u = PYSAT_WCNF()
        for clause in self.problem.clauses:
            lits = [(v + 1) * s for v, s in clause]
            wcnf_u.append(lits, weight=1)
        with PYSAT_RC2(wcnf_u) as rc2:
            model = rc2.compute()
            opt_unweighted = (self.problem.n_clauses - rc2.cost
                              if model is not None else None)

        return opt_unweighted, opt_weighted, time.time() - t0


# Turning raw samples into the final answer

class ResultsProcessor:
    """
    Pipeline:
      1. Identify the raw highest-weight bitstring (diagnostic only).
      2. Project ancilla bits to their CJ2-optimal values given the variable bits
         (vectorised via sign of linear coefficient -- weight-independent).
      3. Compute QUBO energies for each projected bitstring.
      4. Select the best by lexicographic (weighted_sat, -QUBO_energy).
      5. Local search: greedy single-bit flips, accept if weighted_sat strictly
         increases. Re-sync ancillas afterward.
    """

    def __init__(self, problem: WeightedCNFProblem, Q: np.ndarray):
        self.problem = problem
        self.Q = Q

    def count_satisfied(self, bits) -> int:
        return sum(
            1 for c in self.problem.clauses
            if any((s == 1 and int(bits[v]) == 1) or
                   (s == -1 and int(bits[v]) == 0)
                   for v, s in c)
        )

    def weighted_satisfied(self, bits) -> float:
        return sum(
            self.problem.weights[j]
            for j, c in enumerate(self.problem.clauses)
            if any((s == 1 and int(bits[v]) == 1) or
                   (s == -1 and int(bits[v]) == 0)
                   for v, s in c)
        )

    def fix_ancilla_bits(self, bits_str: str) -> str:
        # For each clause, optimal ancilla minimises QUBO contribution.
        # Sign of the linear-in-b_j coefficient decides; coef ~ 0 defaults to 0.
        bits = [int(c) for c in bits_str]
        for j, clause in enumerate(self.problem.clauses):
            (v1, s1), (v2, s2), (v3, s3) = clause
            coef = (-(1 - s1 + s2 * s3)
                    + (-2 * s1) * bits[v1]
                    + (2 * s2) * bits[v2]
                    + (-2 * s3) * bits[v3])
            if coef < -1e-12:
                bits[self.problem.n_vars + j] = 1
            elif coef > 1e-12:
                bits[self.problem.n_vars + j] = 0
            else:
                bits[self.problem.n_vars + j] = 0
        return "".join(str(b) for b in bits)

    def local_search(self, init_bits_str: str) -> np.ndarray:
        # Greedy single-var flip; accept only if weighted satisfaction strictly increases
        n = self.problem.n_vars
        w = self.problem.weights
        var_cls = [[] for _ in range(n)]
        for ci, clause in enumerate(self.problem.clauses):
            seen = set()
            for var, _ in clause:
                if var not in seen:
                    var_cls[var].append(ci)
                    seen.add(var)

        bits = np.array([int(c) for c in init_bits_str], dtype=np.int8)
        sat = np.array([
            any((s == 1 and bits[v] == 1) or
                (s == -1 and bits[v] == 0)
                for v, s in cl)
            for cl in self.problem.clauses
        ], dtype=bool)

        improved = True
        while improved:
            improved = False
            for i in range(n):
                bits[i] ^= 1
                delta, nf = 0.0, {}
                for ci in var_cls[i]:
                    ok = any((s == 1 and bits[v] == 1) or
                             (s == -1 and bits[v] == 0)
                             for v, s in self.problem.clauses[ci])
                    nf[ci] = ok
                    delta += w[ci] * (int(ok) - int(sat[ci]))
                if delta > 1e-12:
                    for ci, ns in nf.items():
                        sat[ci] = ns
                    improved = True
                else:
                    bits[i] ^= 1
        return bits

    def brute_force_dual(self) -> Tuple[Optional[int], Optional[float]]:
        # Exact MAX-SAT optimum by enumeration (n_vars <= 20 only)
        if self.problem.n_vars > 20:
            return None, None
        w = self.problem.weights
        clauses = self.problem.clauses
        best_unw = 0
        best_w = 0.0
        for mask in range(1 << self.problem.n_vars):
            unw, ws = 0, 0.0
            for j, cl in enumerate(clauses):
                ok = any(
                    (sg == 1 and ((mask >> v) & 1) == 1) or
                    (sg == -1 and ((mask >> v) & 1) == 0)
                    for v, sg in cl
                )
                if ok:
                    unw += 1
                    ws += w[j]
            if unw > best_unw:
                best_unw = unw
            if ws > best_w:
                best_w = ws
        return best_unw, best_w

    def qubo_energy(self, bits_str: str) -> float:
        x = np.array([int(c) for c in bits_str], dtype=np.float64)
        return float(x @ self.Q @ x)

    def process_counts(self, raw_counts: Dict) -> Tuple[np.ndarray, np.ndarray,
                                                        np.ndarray, Dict]:
        n_vars = self.problem.n_vars

        # Step 1: diagnostic raw best
        best_raw_str = max(raw_counts.keys(),
                           key=lambda k: self.weighted_satisfied(k))
        best_raw_bits = np.array([int(c) for c in best_raw_str], dtype=np.int8)

        # Step 2: vectorised ancilla projection. The per-clause linear-in-b_j
        # coefficient has a common positive multiplicative factor w_j, so the
        # sign comparison is weight-independent.
        keys = list(raw_counts.keys())
        counts = np.array(list(raw_counts.values()))
        mat = np.array([[int(c) for c in k] for k in keys], dtype=np.int8)

        for j, clause in enumerate(self.problem.clauses):
            (v1, s1), (v2, s2), (v3, s3) = clause
            coefs = (-(1 - s1 + s2 * s3)
                     + (-2 * s1) * mat[:, v1]
                     + (2 * s2) * mat[:, v2]
                     + (-2 * s3) * mat[:, v3])
            mat[:, n_vars + j] = np.where(
                coefs < -1e-12, 1,
                np.where(coefs > 1e-12, 0, 0)
            )

        # Step 3: QUBO energies (float64)
        energies = np.einsum(
            'ij,jk,ik->i',
            mat.astype(np.float64),
            self.Q.astype(np.float64),
            mat.astype(np.float64)
        )

        # Step 4: merge duplicate projected bitstrings
        proj_dict: Dict = {}
        for row, c, e in zip(mat, counts, energies):
            k = "".join(str(b) for b in row)
            if k not in proj_dict:
                proj_dict[k] = {"count": 0, "energy": float(e)}
            proj_dict[k]["count"] += c

        # Step 5: lex-best (weighted_sat first, then lower QUBO energy)
        best_proj = max(
            proj_dict.keys(),
            key=lambda k: (self.weighted_satisfied(k),
                           -proj_dict[k]["energy"])
        )
        best_proj_w = self.weighted_satisfied(best_proj)
        best_proj_bits = np.array([int(c) for c in best_proj], dtype=np.int8)

        # Step 6: weighted local search + re-sync ancillas
        improved = self.local_search(best_proj)
        improved_w = self.weighted_satisfied(improved)
        accept_ls = improved_w > best_proj_w + 1e-12

        if accept_ls:
            synced = self.fix_ancilla_bits("".join(str(b) for b in improved))
            final = np.array([int(c) for c in synced], dtype=np.int8)
        else:
            final = np.array([int(c) for c in best_proj], dtype=np.int8)

        return final, best_proj_bits, best_raw_bits, {
            k: v["count"] for k, v in proj_dict.items()
        }


# CSV logs for results and per-restart convergence

class WCNFLogger:
    def __init__(self, p: int, verbose: bool = False):
        self.p = p
        self.verbose = verbose

    def log_results(self, problem: WeightedCNFProblem,
                    final_bits: np.ndarray,
                    best_proj_bits: np.ndarray,
                    raw_bits: np.ndarray,
                    best_energy: float,
                    best_angles: np.ndarray,
                    opt_time: float,
                    seed: int,
                    opt_unw: Optional[int],
                    opt_w: Optional[float],
                    n_restarts: Optional[int] = None,
                    rc2_unw: Optional[int] = None,
                    rc2_w: Optional[float] = None,
                    rc2_time: float = 0.0) -> None:

        def cnt(bits):
            return sum(
                1 for c in problem.clauses
                if any((s == 1 and int(bits[v]) == 1) or
                       (s == -1 and int(bits[v]) == 0)
                       for v, s in c)
            )

        def wsum(bits):
            return sum(
                problem.weights[j]
                for j, c in enumerate(problem.clauses)
                if any((s == 1 and int(bits[v]) == 1) or
                       (s == -1 and int(bits[v]) == 0)
                       for v, s in c)
            )

        raw_unw, raw_w = cnt(raw_bits), wsum(raw_bits)
        proj_unw, proj_w = cnt(best_proj_bits), wsum(best_proj_bits)
        ls_unw, ls_w = cnt(final_bits), wsum(final_bits)

        alpha_unw = f"{ls_unw / opt_unw:.4f}" if opt_unw else "N/A"
        alpha_w = f"{ls_w / opt_w:.4f}" if opt_w else "N/A"

        csv_file = f"wcnf_results_p{self.p}.csv"
        exists = os.path.exists(csv_file)
        p = self.p
        try:
            with open(csv_file, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if not exists:
                    hdr = ["timestamp_utc", "source", "seed", "p",
                           "n_restarts", "n_vars", "n_clauses",
                           "clause_ratio", "n_qubits", "is_weighted",
                           "raw_sat", "proj_sat", "ls_sat",
                           "raw_wsat", "proj_wsat", "ls_wsat",
                           "opt_clauses", "opt_weight",
                           "rc2_clauses", "rc2_weight",
                           "alpha_unweighted", "alpha_weighted",
                           "total_weight", "best_energy",
                           "opt_time_s", "rc2_time_s"]
                    hdr += [f"g{i}" for i in range(p)]
                    hdr += [f"b{i}" for i in range(p)]
                    w.writerow(hdr)
                row = [datetime.now(timezone.utc).isoformat(),
                       problem.source, seed, p,
                       n_restarts if n_restarts else "Auto",
                       problem.n_vars, problem.n_clauses,
                       f"{problem.ratio:.4f}", problem.n_qubits,
                       problem.is_weighted,
                       raw_unw, proj_unw, ls_unw,
                       f"{raw_w:.2f}", f"{proj_w:.2f}", f"{ls_w:.2f}",
                       opt_unw if opt_unw is not None else "N/A",
                       f"{opt_w:.2f}" if opt_w is not None else "N/A",
                       rc2_unw if rc2_unw is not None else "N/A",
                       f"{rc2_w:.2f}" if rc2_w is not None else "N/A",
                       alpha_unw, alpha_w,
                       f"{problem.total_weight:.2f}",
                       f"{best_energy:.6f}", f"{opt_time:.2f}",
                       f"{rc2_time:.4f}"]
                row += [f"{best_angles[i]:.6f}" for i in range(p)]
                row += [f"{best_angles[p+i]:.6f}" for i in range(p)]
                w.writerow(row)
            print(f"  [Log] Results -> {csv_file}")
        except Exception as e:
            print(f"  [IO WARN] Could not write {csv_file}: {e}")

    def log_convergence(self, eval_log: list, source: str) -> None:
        if not eval_log:
            return
        csv_file = f"wcnf_convergence_p{self.p}.csv"
        exists = os.path.exists(csv_file)
        try:
            with open(csv_file, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if not exists:
                    w.writerow(["source", "restart", "eval", "energy"])
                for row in eval_log:
                    if row:
                        w.writerow([source, row["restart"], row["eval"],
                                    f"{row['energy']:.8f}"])
            print(f"  [Log] Convergence -> {csv_file}")
        except Exception as e:
            print(f"  [IO WARN] {e}")


# Plots: convergence, sample histogram, energy landscape

class WCNFVisualizer:
    def __init__(self, p: int, instance_label: str = ""):
        self.p = p
        self.label = instance_label or "instance"

    def plot_convergence(self, eval_log: list, label: str) -> None:
        if not eval_log:
            return
        min_e = min(r["energy"] for r in eval_log)
        winner = next(r["restart"] for r in eval_log if r["energy"] == min_e)
        rids = sorted(set(r["restart"] for r in eval_log))
        cmap = plt.get_cmap("tab10" if len(rids) <= 10 else "viridis")
        fig, ax = plt.subplots(figsize=(9, 6))
        for i, rid in enumerate(rids):
            trace = [r for r in eval_log if r["restart"] == rid]
            ev = [r["eval"] for r in trace]
            en = [r["energy"] for r in trace]
            iw = (rid == winner)
            ax.plot(ev, en,
                    marker="o" if iw else "",
                    markersize=4 if iw else 0,
                    linestyle="-",
                    color=cmap(i / max(1, len(rids) - 1)),
                    alpha=1.0 if iw else 0.5,
                    linewidth=3.0 if iw else 1.5,
                    label=f"Restart #{rid} {'(BEST)' if iw else ''}")
        ax.set_title(f"WCNF-QAOA Convergence | {label} | p={self.p}", fontsize=12)
        ax.set_xlabel("Evaluation Count")
        ax.set_ylabel("Weighted Energy <H_w>")
        ax.grid(True, linestyle="--", alpha=0.3)
        if len(rids) > 8:
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
        else:
            ax.legend(fontsize=9)
        fname = f"wcnf_convergence_{self.label}_p{self.p}.png"
        _safe_save_fig(fig, fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Plot] Saved {fname}")

    def plot_samples(self, projected_counts: Dict,
                     problem: WeightedCNFProblem, top_k: int = 20) -> None:
        items = sorted(projected_counts.items(), key=lambda x: -x[1])[:top_k]
        labels = [b[:problem.n_vars] for b, _ in items]
        probs = [c / sum(projected_counts.values()) for _, c in items]

        def w_score(b):
            return sum(
                problem.weights[j]
                for j, c in enumerate(problem.clauses)
                if any((s == 1 and int(b[v]) == 1) or
                       (s == -1 and int(b[v]) == 0)
                       for v, s in c)
            )

        scores = [w_score(b) for b, _ in items]
        max_w = problem.total_weight
        norm = plt.Normalize(vmin=0, vmax=max_w)
        cmap = plt.cm.RdYlGn

        fig, ax = plt.subplots(figsize=(max(10, top_k * 0.55), 5))
        bars = ax.bar(range(len(labels)), probs,
                      color=[cmap(norm(s)) for s in scores],
                      edgecolor="black")
        y_pad = max(probs) * 0.04
        ax.set_ylim(0, max(probs) * 1.55)
        for bar, sc in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + y_pad,
                    f"{sc:.0f}/{max_w:.0f}",
                    ha="center", va="bottom", fontsize=7, rotation=45)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right",
                           fontsize=8, fontfamily="monospace")
        ax.set_ylabel("Sampling probability")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, pad=0.01).set_label("Weighted satisfaction")
        fname = f"wcnf_samples_{self.label}_p{self.p}.png"
        _safe_save_fig(fig, fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Plot] Saved {fname}")

    def plot_landscape(self, energy_fn, best_angles: np.ndarray,
                       label: str, n_grid: int = 60,
                       n_qubits: int = 0,
                       zoom: float = 0.0,
                       fine: bool = False) -> None:
        # Default behaviour: cap grid to 25 at >24 qubits unless zoom/fine flags override
        if fine or zoom > 0:
            tg = n_grid
        else:
            tg = min(n_grid, 25) if n_qubits > 24 else n_grid
        mode = "zoom" if zoom > 0 else ("fine" if fine else "default")
        print(f"  [Landscape] Scanning {tg}x{tg} grid "
              f"(weighted energy, mode={mode})...")
        if self.p == 1:
            if zoom > 0:
                gc, bc = best_angles[0], best_angles[1]
                gr = np.linspace(gc - zoom, gc + zoom, tg)
                br = np.linspace(bc - zoom, bc + zoom, tg)
            else:
                gr = np.linspace(-2 * math.pi, 2 * math.pi, tg)
                br = np.linspace(-2 * math.pi, 2 * math.pi, tg)
            GG, BB = np.meshgrid(gr, br)
            E = np.array([[energy_fn(np.array([GG[i, j], BB[i, j]]))
                           for j in range(tg)] for i in range(tg)])
            xl, yl = r"$\gamma$", r"$\beta$"
            og, ob = best_angles[0], best_angles[1]
        else:
            # p > 1: 2D slice around (gamma_0*, beta_0*); other angles held fixed
            gc, bc = best_angles[0], best_angles[self.p]
            gr = np.linspace(gc - 2 * math.pi, gc + 2 * math.pi, tg)
            br = np.linspace(bc - 2 * math.pi, bc + 2 * math.pi, tg)
            GG, BB = np.meshgrid(gr, br)
            E = np.zeros_like(GG)
            for i in range(tg):
                for j in range(tg):
                    pa = best_angles.copy()
                    pa[0], pa[self.p] = GG[i, j], BB[i, j]
                    E[i, j] = energy_fn(pa)
            xl, yl = r"$\gamma_0$ (slice)", r"$\beta_0$ (slice)"
            og, ob = best_angles[0], best_angles[self.p]
        fig, ax = plt.subplots(figsize=(8, 6))
        cf = ax.contourf(GG, BB, E, levels=40, cmap="RdYlBu_r")
        fig.colorbar(cf, ax=ax).set_label("Weighted QAOA Energy <H_w>", fontsize=11)
        ax.scatter([og], [ob], color="white", edgecolors="black",
                   s=120, zorder=5, label="Optimized point")
        ax.set_xlabel(xl, fontsize=13)
        ax.set_ylabel(yl, fontsize=13)
        ax.set_title(f"Weighted Energy Landscape | {label} | p={self.p}", fontsize=11)
        fname = f"wcnf_landscape_{self.label}_p{self.p}.png"
        _safe_save_fig(fig, fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Plot] Saved {fname}")


# Runs the whole pipeline

class WCNFSolver:
    """
    Pipeline: parse -> QUBO -> Ising -> devices -> circuit ->
              classical baselines (RC2 + brute-force) ->
              optimize -> sample -> post-process -> report -> log -> plot.
    """

    def __init__(self, problem: WeightedCNFProblem, config: QAOAConfig):
        self.problem = problem
        self.config = config
        self.best_angles: Optional[np.ndarray] = None
        if problem.n_qubits > 34:
            raise ValueError(
                f"CRITICAL: {problem.n_qubits} qubits exceeds statevector "
                f"limit (34). Reduce n_vars or n_clauses."
            )

    def _print_header(self) -> None:
        p = self.problem
        cfg = self.config
        print(f"\n  {'='*60}")
        print(f"  WCNF-QAOA  |  Weighted CJ2 QUBO")
        print(f"  {'='*60}")
        print(f"  Source     : {p.source}  "
              f"({'WCNF weighted' if p.is_weighted else 'CNF (w=1)'})")
        print(f"  Vars  (n)  : {p.n_vars}")
        print(f"  Clauses(m) : {p.n_clauses}  (ratio alpha={p.ratio:.2f})")
        print(f"  Qubits     : {p.n_qubits}  ({p.n_vars} vars + {p.n_clauses} ancillae)")
        print(f"  Depth p    : {cfg.p}")
        print(f"  Seed       : {cfg.seed}  |  Shots: {cfg.n_shots}")
        print(f"  Backend    : {cfg.backend}")
        if p.is_weighted:
            print(f"  Weights    : {p.weight_summary()}")
            print(f"  TotalWeight: {p.total_weight:.1f}")
        print(f"  {'='*60}")

    def _print_results(self, final_bits: np.ndarray, raw_bits: np.ndarray,
                       best_energy: float, ising_offset: float,
                       best_angles: np.ndarray, opt_time: float,
                       opt_unw: Optional[int], opt_w: Optional[float],
                       rc2_unw: Optional[int] = None,
                       rc2_w: Optional[float] = None,
                       rc2_time: float = 0.0) -> None:
        p = self.config.p
        n_cl = self.problem.n_clauses
        nv = self.problem.n_vars

        def cnt(bits):
            return sum(
                1 for c in self.problem.clauses
                if any((s == 1 and int(bits[v]) == 1) or
                       (s == -1 and int(bits[v]) == 0)
                       for v, s in c)
            )

        def wsum(bits):
            return sum(
                self.problem.weights[j]
                for j, c in enumerate(self.problem.clauses)
                if any((s == 1 and int(bits[v]) == 1) or
                       (s == -1 and int(bits[v]) == 0)
                       for v, s in c)
            )

        sat_unw, sat_w = cnt(final_bits), wsum(final_bits)
        raw_unw, raw_w = cnt(raw_bits), wsum(raw_bits)
        tot_w = self.problem.total_weight

        alpha_unw = f"{sat_unw / opt_unw:.4f}" if opt_unw else "N/A"
        alpha_w = f"{sat_w / opt_w:.4f}" if opt_w else "N/A"

        if int(np.sum(final_bits[:nv])) == 0:
            print(f"\n  [WARN] All-zeros variable bits in final solution.")

        bits_str = "".join(str(b) for b in final_bits)
        print(f"\n{'='*60}\n  WCNF-QAOA Results\n{'='*60}")
        print(f"  Bitstring     : {bits_str[:nv]} | {bits_str[nv:]}")
        print(f"  Variable bits set : {int(np.sum(final_bits[:nv]))} / {nv}")
        print(f"  Best energy   : {best_energy:.6f}"
              f"  (offset-corr: {best_energy + ising_offset:.6f})")
        print(f"  Exec time     : {opt_time:.2f}s")

        print(f"\n  [Raw quantum signal]   "
              f"{raw_unw}/{n_cl} clauses  |  {raw_w:.1f}/{tot_w:.1f} weight")
        print(f"  [Projected + LS]       "
              f"{sat_unw}/{n_cl} clauses  |  {sat_w:.1f}/{tot_w:.1f} weight")

        if rc2_unw is not None:
            print(f"  [RC2 Classical]        "
                  f"{rc2_unw}/{n_cl} clauses  |  "
                  f"{rc2_w:.1f}/{tot_w:.1f} weight  (in {rc2_time:.4f}s)")
        if opt_unw is not None:
            print(f"  [Brute-force OPT]      "
                  f"{opt_unw}/{n_cl} clauses  |  {opt_w:.1f}/{tot_w:.1f} weight")

        print(f"\n  [Approx Ratio]  alpha_unw={alpha_unw}  alpha_w={alpha_w}")

        gamma_str = "  ".join(f"{v:+.6f}" for v in best_angles[:p])
        beta_str = "  ".join(f"{v:+.6f}" for v in best_angles[p:])
        print(f"\n  Optimal gamma : [{gamma_str}]")
        print(f"  Optimal beta  : [{beta_str}]")
        print(f"{'='*60}")

    def run(self) -> None:
        cfg = self.config
        self._print_header()

        # 1. Build weighted QUBO + Ising
        qubo = WeightedCJ2QUBOBuilder(self.problem)
        Q = qubo.Q
        qubo.print_matrix(verbose=cfg.verbose)
        ising = IsingHamiltonian(Q, self.problem.n_qubits)

        # 2. Devices + circuit
        devices = QAOADevices(self.problem.n_qubits, cfg.backend)
        circuit = QAOACircuit(devices, ising, cfg.p, cfg.n_shots, self.problem.n_vars)
        circuit.print_gate_counts()
        if cfg.draw:
            circuit.draw_and_save(self.problem.source)

        # 3. Classical baselines (weighted + unweighted)
        rp = ResultsProcessor(self.problem, Q)
        rc2_proc = RC2Processor(self.problem)
        rc2_unw, rc2_w, rc2_time = rc2_proc.solve()

        opt_unw, opt_w = None, None
        if self.problem.n_vars <= 20:
            opt_unw, opt_w = rp.brute_force_dual()
            if (rc2_unw is not None and opt_unw is not None and rc2_unw != opt_unw):
                print(f"\n  [WARN] RC2/BF unweighted mismatch: "
                      f"RC2={rc2_unw}, BF={opt_unw}. BF is exact.")
            if (rc2_w is not None and opt_w is not None and abs(rc2_w - opt_w) > 1e-6):
                print(f"\n  [WARN] RC2/BF weighted mismatch: "
                      f"RC2={rc2_w:.4f}, BF={opt_w:.4f}. BF is exact.")

        if rc2_unw is not None:
            print(f"\n  Classical OPT : RC2 unw={rc2_unw}/{self.problem.n_clauses}  "
                  f"w={rc2_w:.1f}/{self.problem.total_weight:.1f}  ({rc2_time:.4f}s)")
            if opt_unw is not None:
                print(f"  Brute-force   : unw={opt_unw}/{self.problem.n_clauses}  "
                      f"w={opt_w:.1f}/{self.problem.total_weight:.1f}  (exact cross-check)")
        elif opt_unw is not None:
            print(f"\n  Classical OPT : BF unw={opt_unw}/{self.problem.n_clauses}  "
                  f"w={opt_w:.1f}/{self.problem.total_weight:.1f}")
        else:
            print(f"\n  Classical OPT : N/A (n_vars > 20, RC2 not available)")

        # 4. Optimize
        print(f"\n  [Optimizer] Running weighted p={cfg.p} optimization...")
        optimizer = WarmStartOptimizer(circuit, cfg,
                                       self.problem.n_vars, self.problem.n_clauses)
        t0 = time.time()
        best_energy, best_angles, eval_log = optimizer.run()
        opt_time = time.time() - t0

        if best_angles is None:
            print("  [WARN] All restarts failed. Defaulting to zero angles.")
            best_angles = np.zeros(2 * cfg.p)
        self.best_angles = best_angles

        # 5. Sample + post-process
        print(f"\n  [Sampling] {cfg.n_shots} shots executing...")
        raw_counts = circuit.sample(best_angles)
        final_bits, best_proj_bits, raw_bits, proj_counts = rp.process_counts(
            raw_counts
        )

        # 6. Report
        self._print_results(final_bits, raw_bits, best_energy, ising.offset,
                            best_angles, opt_time, opt_unw, opt_w,
                            rc2_unw=rc2_unw, rc2_w=rc2_w, rc2_time=rc2_time)

        # 7. Log + plot
        logger = WCNFLogger(cfg.p, cfg.verbose)
        logger.log_results(self.problem, final_bits, best_proj_bits, raw_bits,
                           best_energy, best_angles, opt_time, cfg.seed,
                           opt_unw, opt_w, n_restarts=cfg.n_restarts,
                           rc2_unw=rc2_unw, rc2_w=rc2_w, rc2_time=rc2_time)
        logger.log_convergence(eval_log, self.problem.source)

        instance_label = os.path.splitext(self.problem.source)[0]
        viz = WCNFVisualizer(cfg.p, instance_label)
        viz.plot_convergence(eval_log, self.problem.source)
        if cfg.plot_landscape:
            viz.plot_landscape(circuit.energy, best_angles, self.problem.source,
                               cfg.landscape_grid, self.problem.n_qubits,
                               zoom=cfg.landscape_zoom, fine=cfg.landscape_fine)
        if cfg.plot_samples:
            viz.plot_samples(proj_counts, self.problem)


# Command line

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WCNF-QAOA: Weighted CJ2 QUBO for Weighted Partial MAX-3-SAT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python src/qaoa_wcnf_mod.py instances/office_schedule_30q.wcnf \\\n"
            "      --p 1 --shots 65536 --seed 42 \\\n"
            "      --plot-landscape --plot-samples --backend gpu\n"
        ),
    )
    parser.add_argument("cnf_file", help="Path to DIMACS .cnf or .wcnf file.")
    parser.add_argument("--p", type=int, default=1, help="QAOA depth. Default 1.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed. Default 42.")
    parser.add_argument("--shots", type=int, default=65536, help="Sampling shots. Default 65536.")
    parser.add_argument("--backend", default="cpu", choices=["cpu", "gpu"],
                        help="Device backend. Default cpu.")
    parser.add_argument("--n-restarts", type=int, default=None,
                        help="Optimizer restarts. Default auto = max(8, 6+2p).")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose: stagnation warnings + full QUBO dump.")
    parser.add_argument("--draw", action="store_true",
                        help="Save circuit diagram (wcnf_circuit_p{p}.png).")
    parser.add_argument("--plot-samples", action="store_true",
                        help="Save bitstring histogram (wcnf_samples_*.png).")
    parser.add_argument("--plot-landscape", action="store_true",
                        help="Save energy landscape (wcnf_landscape_*.png).")
    parser.add_argument("--landscape-grid", type=int, default=60,
                        help="Landscape grid resolution. Default 60.")
    parser.add_argument("--landscape-zoom", type=float, default=0.0,
                        help="Zoom radius around (gamma*, beta*). Default 0 (full).")
    parser.add_argument("--landscape-fine", action="store_true",
                        help="Bypass the 25-grid cap at n_qubits > 24.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if not os.path.isfile(args.cnf_file):
        sys.exit(f"File not found: {args.cnf_file}")

    problem = WeightedCNFProblem.from_file(args.cnf_file)
    print(f"  Loaded: {problem}")

    config = QAOAConfig(
        p=args.p,
        seed=args.seed,
        n_shots=args.shots,
        backend=args.backend,
        n_restarts=args.n_restarts,
        draw=args.draw,
        plot_samples=args.plot_samples,
        plot_landscape=args.plot_landscape,
        landscape_grid=args.landscape_grid,
        landscape_zoom=args.landscape_zoom,
        landscape_fine=args.landscape_fine,
        verbose=args.verbose,
    )

    WCNFSolver(problem, config).run()


if __name__ == "__main__":
    main()
