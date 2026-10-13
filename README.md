# Be: which radial freedom moves the node

Variational and fixed-node diffusion Monte Carlo for the beryllium atom
(Z = 4, four electrons), written in JAX. The study is built around one
question: of the radial freedoms in a compact multi-determinant ansatz,
which ones change the nodal surface, and by how much.

The design is a set of wave functions that differ by one construction
element at a time, so that a difference in the fixed-node energy can be
attributed to that element rather than to the overall quality of the
ansatz.

## Result

Grouping the models by construction, not by any fitted quantity:

| group | models | E_FN − exact (mHa) | chi2 |
|---|---|---|---|
| 1s radial modulation optimised | Q, Q', L, Q+J1, D, D+J1 | +0.137(16) | 7.3/5 |
| 1s radial modulation fixed to unity | U, U+J1, F+J1 | +0.335(28) | 2.3/2 |
| difference | | **0.198(32)** | 6.1 sigma |

At fixed Jastrow the same separation appears pairwise: adding 2s and 2p
modulation to U+J1 gives −0.02(8) mHa, while adding the 1s modulation to
F+J1 gives −0.20(6) mHa (3.2 sigma). The exponent alpha_1s is fixed to Z
by the electron-nucleus cusp, so m_1s is the only 1s radial freedom; the
node depends on the ratios phi_2s/phi_1s and phi_2p/phi_1s.

Reference energies, defined once in `src/be_nqs/wavefunction.py`:
E_HF = −14.573023, E_exact = −14.667356 Eh.

## Per-model numbers

E_FN at dtau = 0.0025, as E_FN − exact in mHa:

| model | internal name | E_FN − exact | DMC seed |
|---|---|---|---|
| D | cas2 | +0.016(69) | 2051 |
| Q+J1 | quadJ1 | +0.087(41) | 2061 |
| Q | quadA | +0.119(37) | 2031 |
| Q' | quadB | +0.152(33) | 2041 |
| D+J1 | full | +0.155(35) | 2067 |
| L | lin | +0.187(35) | 2043 |
| F+J1 | fix1s | +0.290(49) | 2065 |
| U+J1 | nomodJ1 | +0.311(58) | 2063 |
| U | nomod | +0.383(43) | 2031 |

Q and U share DMC seed 2031; their block correlation is rho = +0.04 and
their difference is −0.264(57) mHa.

VMC, where recorded in the final run:

| model | internal | E_VMC | Var | corr |
|---|---|---|---|---|
| D+J1 | full | −14.66697(12) | 0.0086 | 99.59(13) % |
| Q+J1 | quadJ1 | −14.66689(11) | 0.0086 | 99.51 % |
| F+J1 | fix1s | −14.66676(14) | 0.0115 | |
| D | cas2 | −14.66663(12) | 0.0107 | 99.23 % |
| D_U | cas2rigid | −14.66647(13) | 0.0134 | |
| U+J1 | nomodJ1 | −14.66633(14) | 0.0122 | |

The Jastrow accounts for most of the VMC error in every model: 98 % for
D, 81 % for Q+J1, 80 % for Q, 60 % for D+J1, 51 % for F+J1.

## Files

The five modules are a library; the two entry points are in `scripts/`.

| file | contents |
|---|---|
| `src/be_nqs/wavefunction.py` | VMC and the wave-function definition. Single source for the reference energies. `log_psi.parts` separates determinant and Jastrow; `radial_profiles` and `radial_diagnostics` export the orbital shapes; `freeze=` holds parameters fixed; `one_body=True` adds a one-body Jastrow (MLP 1→8→1, SiLU, on u = ln(1+t), with w'(0) = 0). Models without it are bit-identical to the earlier pipeline. |
| `src/be_nqs/dmc.py` | Fixed-node DMC: importance sampling, drift cap, Metropolis correction, continuous weights with comb resampling, population control. |
| `src/be_nqs/nodes.py` | Nodal diagnostics: sign and mismatch counts from independent chains, radial weights w_l. |
| `src/be_nqs/stats.py` | Blocking, chi2 and group fits. |
| `src/be_nqs/figures.py` | Figures. |
| `scripts/run_study.py` | Driver. Stages: `validate`, `vmc`, `nodes`, `dmc`, `extra`, `cas`, `j1`, `popbias`, `figures`, `summary`. |
| `scripts/check_invariants.py` | Read-only. Scans r* for the contracted mixing coefficient and computes the density-weighted matched gauge relative to Q, with radial residuals. |
| `paper/NQS.tex` | Manuscript. `paper/NQS.pdf` is the compiled version. |

## Installation

```bash
python -m pip install -e .          # Python >= 3.11
```

JAX is installed as a plain CPU wheel by the line above. For GPU or for
Apple-silicon acceleration, install the right JAX wheel first, following
the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html),
then run the editable install.

`requirements.txt` pins the exact versions the published numbers were
produced with (Python 3.11.15, JAX 0.10.2). The lower bounds in
`pyproject.toml` are those same versions: older releases are untested,
not unsupported.

## Running

```bash
python scripts/run_study.py validate      # hydrogen, nodeless
python scripts/run_study.py vmc           # optimise all models
python scripts/run_study.py dmc           # FN-DMC at dtau = 0.0025
python scripts/run_study.py summary       # tables
```

The full sequence took 581 minutes in the run the numbers above come
from. `vmc` writes one `.npz` per model holding the parameters, the
equilibrated walkers and the clip bounds; the bounds are part of the
wave-function definition, since a parameter sitting on a bound defines a
different function if the bound moves.

VMC seeds: Q uses 42, Q' uses 7. Q and Q' differ only by that seed and
serve as the replica pair.

## Reproducibility

Generated files are not in the repository. The pipeline writes to two
directories, both empty here:

    data/      .npz: trial functions, DMC runs, nodal analyses
    results/   figures and the summary tables

Every stage skips outputs that already exist (`--force` to redo them), so
an interrupted run can simply be relaunched. Reproducing the tables above
from nothing means running the stages in order:

```bash
for s in validate vmc nodes dmc extra cas j1 popbias figures summary; do
    python scripts/run_study.py $s
done
```

`--quick` shrinks every run to a few seconds, enough to check that the
pipeline works end to end. Quick outputs carry the same file names as
production ones, so redirect them:

```bash
BE_NQS_DATA=/tmp/quick BE_NQS_RESULTS=/tmp/quick \
    python scripts/run_study.py validate --quick
```

Both variables default to `data/` and `results/` next to this file.

## What this repository does not claim

These were considered and either failed or were not established. They
are listed because several of them appeared in earlier drafts.

- **E_FN is not linear in the contracted mixing coefficient.** The
  apparent linearity depended on the cutoff radius r*: the slope is
  positive for r* = 0.5–1.25 bohr and negative for r* >= 1.5, and the
  value that gave the best chi2 had been picked by hand. In the matched
  gauge the slope is zero within the modulated models.
- **No claim that nine nodal parameters match the 191 of a backflow
  ansatz.** Retracted.
- **No claim to improve on Seth et al. in VMC.** D+J1 is +0.22(12) mHa
  above it, 1.8 sigma; it is comparable to Morales et al. at −0.09(13).
- **D is not shown to have the exact node** despite +0.016(69) mHa.
- The raw mixing coefficient does not order E_FN (chi2 46.8/7). E_VMC
  orders it weakly (Spearman 0.57, p = 0.11); the variance does better
  (0.81, p = 0.008).
- The Jastrow-independence test is not used: with the determinant alone
  the Padé cusp is retained, so the test is weak (+1.051(812) mHa).
- Validation is hydrogen only, and hydrogen is nodeless.
- Branching uses dtau rather than an effective time step, so the
  algorithm is not the Umrigar–Nightingale–Runge scheme; only the drift
  cap follows it.
- A constant fit to the time-step series is not excluded (chi2 6.81/3,
  p = 0.08). E_FN is quoted at dtau = 0.0025 with the spread of the
  linear and quadratic fits, −0.01 to −0.05 mHa, as a systematic.
- The time-step series was run for Q only; equal bias across models is
  an assumption.
- Some parameters sit on their bounds: m_1s c2 in Q+J1 and D+J1, both
  m_3d coefficients in D and D+J1, m_2s c1 in F+J1.
- The single-determinant series S0–S3 was produced by the earlier
  pipeline and has not been rerun here. That pipeline is not part of this
  repository, so `comparison_jastrow.pdf` (Fig. 2 of the manuscript)
  cannot be regenerated from the code published here and compiles as a
  placeholder.
- The radial residual used to group the models is reference-dependent
  and descriptive; it is not a gauge-independent distance between nodes.

## State of the manuscript

`paper/NQS.tex` compiles with `latexmk -pdf NQS.tex` and is included
compiled as `paper/NQS.pdf` (11 pages). Two things in it are unfinished,
and the compiled PDF shows them as framed placeholders rather than
hiding them:

- `fig_node_groups.pdf` — never produced. It would show E_FN of the nine
  trial functions grouped by the treatment of the 1s orbital, with the
  two weighted means of the group table as bands.
- `comparison_jastrow.pdf` — belongs to the earlier pipeline, which is
  not published here (see above).

The 16 entries of the inline bibliography are still `TODO` stubs: author
and year are there, the full references are not.

## Not published here

An `OLD/` directory holds two earlier generations of this pipeline and
the September manuscript. It is deliberately excluded: it is three
near-duplicate copies of the same modules, and nothing in the results
above depends on it.

## Open

- A gauge-independent measure of the distance between two nodal
  surfaces.
- Widen the upper bound on the 1s quadratic coefficient, or parametrise
  the radial ratios directly, and measure E_FN against the 1s shape.
- D has 3.4 times the DMC block variance of Q, against 1.13 in VMC.
  Unexplained.

## Licence and citation

MIT, see [LICENSE](LICENSE). If you use this code, `CITATION.cff` holds
the metadata GitHub's *Cite this repository* button reads.
