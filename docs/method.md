# How it works

The short version of what happens between a file of rules and an answer you can
trust.

## The problem

Weighted Partial MAX-3-SAT. Some clauses are hard and have to be satisfied, so
they carry a huge weight. The rest are soft and carry normal weights. You want
whichever assignment adds up to the most satisfied weight.

Input is plain DIMACS. If the header says `p wcnf`, the weights are read from
the file. If it says `p cnf`, every clause just gets weight 1.

## Turning it into something a quantum circuit can handle

Each clause becomes an energy penalty, so the lowest energy bitstring is the
best answer. I used the CJ2 encoding from Ansótegui and Levy (2024) for this.
It adds one extra variable per clause, and in exchange the whole thing stays
quadratic.

That's the reason to bother: a quadratic problem (a QUBO) maps straight onto
two-qubit gates. Encoding 3-SAT directly would give you cubic terms and a much
worse circuit.

## The circuit

QAOA alternates a cost layer and a mixer layer, p times over. Each layer has
two parameters, gamma and beta. On everything I tested, p = 1 was enough to
reach the classical optimum.

It runs on PennyLane, either `lightning.qubit` on CPU or `lightning.gpu` with
cuStateVec on an NVIDIA card. Memory is what stops you going bigger, because the
statevector doubles every time you add a qubit. At 30 qubits one statevector is
16 GiB, and the solver keeps two (an exact one for optimising and one for
sampling), so a run needs at least 34 GB.

## Finding good parameters

Picking gamma and beta by hand doesn't scale, so the search does it in stages:

1. A grid scan, started from the structure of the SAT problem
2. Gaussian perturbation around whatever looked best
3. Latin Hypercube Sampling, to cover more ground without a dense grid

Each candidate then gets refined with L-BFGS-B. The gradients are exact, using
adjoint differentiation rather than parameter shift, which keeps memory sane.
There's also a guard that restarts the optimiser when it stops improving.

## Checking the answer

Nothing gets reported until something independent agrees with it:

- RC2 MaxSAT, an exact classical MaxSAT solver, through python-sat
- Brute force, on instances small enough to enumerate
- The heuristics in `classical_baselines.py`: greedy hill-climbing, simulated
  annealing and random sampling, which tell you how hard the instance really is

If QAOA and the classical solvers ever disagree, look at the encoding first.

## Reading the output

- `alpha_w` is satisfied weight over the optimum. 1.0 means you got the best
  possible answer.
- The bitstring histogram shows how often each answer came up. A sharp peak on
  the optimum is good. A flat one means the parameters were poor.
- The energy landscape plots cost against gamma and beta, which is how you tell
  whether the optimiser ever had a chance.

## References

- Farhi, Goldstone and Gutmann (2014). A Quantum Approximate Optimization
  Algorithm. arXiv:1411.4028
- Ansótegui and Levy (2024). SAT, Gadgets, Max2XOR, and Quantum Annealers.
  arXiv:2403.00182
- Rodríguez-Farrés, Ballester, Ansótegui, Levy and Cerquides (2024).
  Implementing 3-SAT Gadgets for Quantum Annealers with Random Instances.
  ICCS 2024, pp. 277-291. doi:10.1007/978-3-031-63778-0_20
