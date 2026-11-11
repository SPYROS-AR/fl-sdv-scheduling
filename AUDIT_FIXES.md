# Audit fixes

Code changes requested by the audit of November 2026 (Section 5 of the audit report),
on top of commit `b6b9d63`. Default settings reproduce the decision rules of the
submitted runs; every new behaviour is opt-in except the WAIT bug fix and seeding.

## 1. Changes

| Audit item | Change | Where |
|---|---|---|
| 5.1 Deterministic trace and seeds | The server broadcasts each round's trace vector and vehicle *i* reads entry *i*, so vehicle *i* in round *r* always gets sample `(r-1)*N + i`, the same per-round sample set as before, now assigned by partition id instead of Flower's random client order. `--seed` seeds the initial model (created on the server instead of fetched from a random client), the data order and dropout of every (vehicle, round, epoch), and the order in which over-capacity uploads are dropped. Output names get `_s<seed>`. | `server.py`, `client.py`, `scheduler.py`, `main.py` |
| 5.2 WAIT bug | A WAIT no longer discards a vehicle's pending CONTINUE trajectory: the weights stay on disk and the round counts towards the staleness limit `K_max`. | `client.py` |
| 5.3 Utility as the stopping rule | `--stop-rule utility` stops when the predicted gain is below `g_min = (tau + beta*U_crit/rho + gamma*D) / delta`, so the threshold rises with safety load and drift. `tau` defaults to 0.75 for this rule, which places `g_min` near the old `G_th = 0.03` at typical load (U_crit = 0.4, D = 2). `alpha` is documented as accounting-only. | `policy.py`, `main.py` |
| 5.4 Per-vehicle log | `clients_<run>.csv`: one row per vehicle and round (state, decision, reason, epochs, link, load, drift, gain, costs, simulated compute and upload times). | `server.py` |
| 5.5 Simulated V2X clock | Round log columns `uplink_airtime_s`, `uplink_straggler_s`, `sim_round_time_s`, `sim_time_s` (definitions below). | `server.py` |
| 5.6 Baselines and ablations | `fixedcap_E`, `slackfill`, `threshold_E`, `randpart_E_P`, `dynamic_nostop`, `dynamic_nocontinue`; any name plus `_fedprox`. | `policy.py`, `client.py`, `scheduler.py` |
| 5.7 Procedure | `scripts/aggregate_results.py` (multi-seed means, 95% CIs, last-k accuracy, uplink to sustain a target, LaTeX table, plots); `scripts/submit_seeds.sh` (SLURM array: 14 schemes x 3 alphas x 5 seeds). | `scripts/` |
| 5.8 Model options | `--trace-mode sticky` (each vehicle stays in a road environment for a geometric dwell time), `--ucrit-offset` (stress test: U_crit + offset, capped at 0.95), `--upload-deadline-s` (link-dependent drops). The log records samples processed and validation passes. | `server.py`, `main.py` |
| 5.9 Hygiene | Decision logic moved to the PyTorch-free `src/policy.py`; dead `if epoch == 0` removed; tests in `tests/`; `.gitignore`. | |

## 2. New finding: the legacy gain estimate is biased upwards on a rising loss

The fitted curve `L_inf + a*exp(-b*e)` can only decrease. On a flat or rising (over-fitting)
validation loss the fit therefore regresses to the mean, and the legacy estimate
`G_hat = H[e] - L_hat(e+1)` reports a spurious gain: 0.10 for the smoothed history
`[1.0, 1.1, 1.2]`, more than three times `G_th`. In a simulation of the first stopping
decision (raw loss rising by 0.05 per epoch, noise s.d. 0.02, 200 trials), the legacy
estimate exceeded `G_th` in 12% of the trials, so the engine kept training while the loss
was going up. `--gain-estimate curve` uses the slope of the fitted curve,
`f(e) - f(e+1)`. It is close to the legacy estimate on a decaying loss and about zero on a
flat or rising one (0% of the same trials). The legacy estimate remains the default.
`tests/test_policy.py` pins both behaviours.

## 3. Compatibility with the submitted runs

Identical with the default (legacy) settings, verified by tests:
- every decision rule (admission, stopping, transmission), compared with a line-by-line
  transcription of `client.py` at `b6b9d63` on randomized inputs;
- the curve-fit prediction (identical outputs on 300 random loss histories);
- the set of trace samples per round, and the capacity-drop model (ten uploads per round for
  250 rounds give the 567 drops of every fixed baseline);
- the cost and objective accounting, and all existing CSV columns (new columns are inserted
  before `round_time_s`; read the files by column name).

Different by design: the WAIT fix, the assignment of samples to vehicles, the seeded random
number generators, and the server-side initial model. New runs will therefore not
reproduce the old numbers exactly. **Rerun every scheme with this code before updating the
paper, and do not mix old and new CSV files.**

## 4. Experiment names and options

| Name | Scheme |
|---|---|
| `dynamic` | Policy engine (as submitted) |
| `dynamic_nostop` | Ablation: trains the whole slack budget; same WAIT/SYNC/CONTINUE rules |
| `dynamic_nocontinue` | Ablation: a weak link means WAIT, never CONTINUE |
| `fixed_E`, `fedprox_E` | E epochs every round, ignoring the slack (as submitted) |
| `fixedcap_E` | `min(E, floor(rho*(1-U_crit)))` epochs; WAIT if that is 0 |
| `slackfill` | `floor(rho*(1-U_crit))` epochs every round ("more epochs = better"); always uploads |
| `threshold_E` | Slack-capped E epochs, but only when `B >= B_th`; otherwise WAIT |
| `randpart_E_P` | E epochs for a random P% of the vehicles each round (deterministic per seed) |
| `<name>_fedprox` | Same scheme with the FedProx proximal term (`--fedprox-mu`) |

Main options (all optional): `--experiments`, `--seed`, `--stop-rule {legacy,utility}`,
`--gain-estimate {last,curve}`, `--tau-continue`, `--bw-threshold`, `--gain-threshold`,
`--drift-threshold`, `--max-continue-rounds`, `--trace-mode {iid,sticky}`,
`--sticky-dwell`, `--ucrit-offset`, `--upload-deadline-s`, `--sim-samples-per-s`,
`--num-vehicles`, `--tag` (use the `--tag=-rev` form, since a leading `-` is otherwise read
as an option; the tag is appended to the experiment name in output files), and
`--synthetic-data` (random images, for smoke tests only).

Simulated clock. For an uploading vehicle, `t_up = 8.389 * M / B` seconds (M in MiB, B in
Mbit/s). Compute time is `t_comp = (e*n_train + val_passes*n_val/3) / (S*(1 - U_crit))`,
with S training samples per second on the whole processor (`--sim-samples-per-s`,
default 1000, **an assumption to calibrate or vary**). A round lasts
`max_i (t_comp + t_up)`, a synchronous barrier over all vehicles that train;
`uplink_straggler_s` is the slowest upload and `uplink_airtime_s` the sum of upload times.

Notes on configuration:
- With the paper's weights (beta = 1, gamma = 0.006, delta = 27), `g_min` only ranges over
  about 0.029-0.035 on the trace, so U_crit and drift barely affect stopping. For them to
  matter, beta must grow (beta = 10 roughly doubles `g_min` at U_crit = 0.8). Tune such
  weights on a held-out setting (e.g. alpha_Dir = 0.2 and a separate seed), not on the
  evaluation runs.
- `sticky` gives persistence at the level of road environments only: within an environment,
  successive samples are still independent.

## 5. Tests

`pip install pytest` (or `requirements-dev.txt`), then `pytest tests` from the repository
root: 72 tests, about 30 s on a CPU.
- `test_policy.py`: equivalence with the submitted logic; the redundancy of the legacy utility
  test (`Delta U >= 0.595 > tau` whenever `G_hat >= 0.03` and `D <= 20`); the gain-estimate
  bias; admission, staleness, baselines, seeding.
- `test_server.py`: trace mapping (iid and sticky); the 567-drop reproduction; drop selection
  independent of result order; deadline drops; CSV columns; simulated clock.
- `test_client.py`: real CPU training on tiny synthetic data; the WAIT fix across a simulated
  Ray actor restart; reproducible seeds; every baseline and ablation; both stop rules with
  FedProx.
- `test_cli.py`: experiment names, `main.py` wiring of seeds, tags and stop rules, and the
  aggregation script, which must reproduce the numbers of the revised paper from
  `final_metrics/`.

End-to-end smoke tests (Flower 1.33 simulation with Ray, CPU, synthetic images):

```bash
python src/main.py --synthetic-data --num-vehicles 4 --rounds 3 --seed 1 --job-id smokeA \
  --gpus_per_client 0 --experiments dynamic_fedprox fixed_2 fedprox_2 fixedcap_10 slackfill \
  threshold_2 randpart_2_50 dynamic_nostop dynamic_nocontinue
python src/main.py --synthetic-data --num-vehicles 4 --rounds 4 --seed 2 --job-id smokeB \
  --gpus_per_client 0 --experiments dynamic fixed_2 --trace-mode sticky --sticky-dwell 3 \
  --upload-deadline-s 3.0 --stop-rule utility --gain-estimate curve --ucrit-offset 0.25 --tag=-rev
```

In both runs every logged trace value and decision matched the scheme definitions and an
independently rebuilt trace. In the second run, the U_crit offset triggered the
compute-starvation WAIT, which never fired in the submitted runs, and every upload slower
than the deadline was dropped.

## 6. Running the new evaluation

```bash
pytest tests                                   # 1. unit and integration tests
sbatch scripts/submit_seeds.sh                 # 2. 210 tasks; or a subset, e.g. --array=0-41
python scripts/aggregate_results.py <dir> --targets 70 75 --plot --out results/   # 3.
```

`results/summary.csv` and `results/table_results.tex` hold mean +/- 95% CI per scheme and
alpha (last-10-round accuracy, uplink MiB, drops, compute interference, drift per update,
uplink needed to sustain each target, simulated times, objective value). `--plot` draws
accuracy against cumulative uplink with CI bands (shown from three seeds on).
`src/visualization.py` is unchanged; it reads one file per scheme and alpha, without a seed
suffix.

## 7. Not done here

- CIFAR-10 runs: no GPU or Hugging Face access in the environment where this was written.
  All numbers in the paper must come from new runs.
- A FedNova baseline (Flower's baselines repository has one).
- A per-vehicle trajectory model for the channel and the safety load.
- A new hyperparameter sweep on the current code.
