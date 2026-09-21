# qaoa-maxsat

A QAOA solver for Weighted Partial MAX-3-SAT, with classical solvers to check
its answers against.

## What the problem is

You have some rules that must hold (hard clauses) and some you'd like to hold
if possible (soft clauses). Each rule has a weight. The job is to find the
true/false assignment that satisfies the most weight.

The main example is an office schedule: which days each of 3 teams comes in.
"Someone has to be in every day" is a hard rule. "Operations would rather be in
mid-week" is a soft one.

## What this does

It reads a problem in DIMACS `.cnf` or `.wcnf` format, turns it into a QUBO
using the CJ2 encoding, and runs QAOA on PennyLane. It can run on your CPU or
on an NVIDIA GPU. It tunes the circuit parameters itself, so you don't have to
guess them, and then checks the answer against classical solvers before it
reports anything.

## What I got out of it

The main run was that office schedule: 7 people in 3 teams over a 5-day week,
with 10 hard rules and 5 soft ones. It has 15 variables, and CJ2 adds one qubit
per clause, so it runs on 30 qubits. At depth p = 1 it found the best possible
answer, 1050 out of 1050, and RC2 MaxSAT and brute force both agree. It ran on
a single NVIDIA H100 NVL (94 GB), and one run takes about 50 minutes.

I also ran three 30-qubit instances from my drug safety work. Each one is a
patient profile with 12 candidate drugs, where the solver has to follow
treatment guidelines while avoiding three-drug combinations flagged as risky.

## Running it

```bash
git clone https://github.com/Ashlar-30/qaoa-maxsat.git
cd qaoa-maxsat
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Start with a small one on CPU. It takes about 15 seconds:

```bash
python src/qaoa_wcnf_mod.py instances/Fifteen.cnf --p 1 --shots 65536 --seed 42
```

The full 30-qubit schedule needs an NVIDIA GPU with at least 34 GB of memory.
I used an H100 NVL with 94 GB:

```bash
pip install "pennylane-lightning[gpu]" custatevec-cu12
python src/qaoa_wcnf_mod.py instances/office_schedule_30q.wcnf --p 1 --shots 65536 --seed 42 --backend gpu
```

To check an answer with the classical solvers (a few seconds, no GPU needed):

```bash
python src/classical_baselines.py instances/office_schedule_30q.wcnf
```

Output files all start with `wcnf_` and go into whatever folder you run from.

## Options you'll actually use

`--p` is the QAOA depth, `--shots` is how many measurements per sample,
`--seed` fixes the randomness so runs repeat, and `--backend` picks cpu or gpu.
There's also `--plot-landscape`, `--plot-samples` and `--draw` if you want
figures out of it.

`python src/qaoa_wcnf_mod.py --help` lists the rest.

## What's in here

- `src/` the solver and the classical baselines
- `instances/` the office schedule, the three patient profiles, and some small
  test problems (10 to 20 qubits)
- `docs/` how it works, and how to reproduce the runs

## What it doesn't do

This is research code, not a library. QAOA at p = 1 matches brute
force here, it doesn't beat it, and it was never meant to. The point was to get
a correct pipeline that someone else can run and check.

Instance size is capped by memory, because the statevector doubles with every
qubit you add. 30 qubits needs at least 34 GB, and that's the biggest size in
here.

## Licence

MIT, see [LICENSE](LICENSE).

## Me

Ashwin Kumar Baskaran
