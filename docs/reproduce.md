# Reproducing the runs

Everything is seeded, so the same seed on the same machine gives the same
numbers.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you're using a GPU, check it's actually visible first:

```bash
nvidia-smi
python -c "import pennylane as qml; qml.device('lightning.gpu', wires=4); print('GPU OK')"
```

## Start small

```bash
python src/qaoa_wcnf_mod.py instances/Fifteen.cnf --p 1 --shots 65536 --seed 42
```

This takes about 15 seconds on a laptop CPU. Do this before anything bigger, so
you know the install is fine.

## With plots

```bash
python src/qaoa_wcnf_mod.py instances/3p3d.cnf \
    --p 1 --shots 65536 --seed 42 \
    --plot-landscape --plot-samples --backend gpu
```

## The 30-qubit runs

These need a GPU with at least 34 GB of memory. I ran them on a single NVIDIA
H100 NVL (94 GB HBM3) with PennyLane 0.44.1 and cuStateVec 1.13.0, where the
office schedule takes about 50 minutes.

```bash
python src/qaoa_wcnf_mod.py instances/office_schedule_30q.wcnf --p 1 --shots 65536 --seed 42 --backend gpu
python src/qaoa_wcnf_mod.py instances/profile1_ra_cvd_30q.cnf --p 1 --shots 65536 --seed 42 --backend gpu
python src/qaoa_wcnf_mod.py instances/profile2_post_mi_30q.cnf --p 1 --shots 65536 --seed 42 --backend gpu
python src/qaoa_wcnf_mod.py instances/profile3_severe_ra_30q.cnf --p 1 --shots 65536 --seed 42 --backend gpu
```

The optimum for each is 1050, 600, 630 and 670.

## Check it against the classical solvers

```bash
python src/classical_baselines.py instances/office_schedule_30q.wcnf instances/profile1_ra_cvd_30q.cnf instances/profile2_post_mi_30q.cnf instances/profile3_severe_ra_30q.cnf
```

This runs in a few seconds on any machine. The brute-force optimum it prints
should match what QAOA found. If it doesn't, check the encoding first.

## Things worth knowing

`--seed` controls all the randomness. Change it to see whether a result is
stable, not to go fishing for a better one.

GPU and CPU runs can differ in the last few decimal places because of
floating-point ordering.

The 30-qubit instances need at least 34 GB of GPU memory, because the solver
keeps two 16 GiB statevectors (one exact, one for sampling). Below that they
won't run at all.

## Not done yet

- A smoke test that runs the smallest instance and checks the known optimum
- CI to run that test on every push
