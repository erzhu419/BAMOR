# BAMOR Server Deployment Package

Runs the BAMOR vs. baselines comparison on the synthetic multi-objective
environment. Produces a JSON per (method, seed) and a final aggregated
mean±std table over HV overall + per-mode HV.

## Contents
- `code/`                 — experiment code (env, BAMOR, training, evaluation)
- `run_server.sh`         — auto-resource one-liner launcher
- `requirements.txt`      — Python dependencies
- `results/`              — output directory (created on first run)

## Requirements
- Python 3.10+
- CUDA-capable PyTorch (tested with torch 2.4 + CUDA)
- `nvidia-smi` available on PATH
- ~1.5 GB free GPU memory per parallel job

Install deps (inside your conda env):
```
pip install -r requirements.txt
```

(Optional) for CAPQL baseline:
```
pip install pymoo mo-gymnasium
# plus morl-baselines (from https://github.com/LucasAlegre/morl-baselines)
```
If any of these are missing, CAPQL is silently skipped.

## Run everything (one command)
```
bash run_server.sh
```

The launcher:
1. Auto-detects free memory per GPU via `nvidia-smi` — picks a GPU with
   `≥ MEM_PER_JOB_MB + SAFETY_MARGIN_MB` free before launching each job.
2. Caps CPU parallelism at `min(nproc/2, 8)` by default.
3. Schedules 5 methods × 5 seeds = 25 jobs (plus CAPQL if installed).
4. Polls every `POLL_SEC` (default 20 s) when no resources are free.
5. Skips a (method, seed) if its result JSON already exists (safe to re-run).
6. Aggregates `results/server_<ts>/*_seed*.json` into a summary table at end.

## Environment overrides
```
TOTAL_STEPS=100000 SEEDS="0 1 2" METHODS="bamor uniform" \
MEM_PER_JOB_MB=2000 MAX_CPU_PARALLEL=4 \
bash run_server.sh
```

## Output
Each job writes `results/server_<ts>/<method>_seed<N>.json` with:
- `hv_overall`       — global Pareto front HV
- `hv_per_mode[i]`   — per-mode evaluation HV (mode 0/1/2)
- `hv_curve`         — training HV trajectory
- `mode_switches`    — detected mode change points (BAMOR only)

Logs are in `results/server_<ts>/logs/<method>_seed<N>.log`.

## Expected results (from the 100K-step pilot on local GPU)
```
  Method       HV overall     HV Mode0     HV Mode1     HV Mode2
  bamor        130920±16860   21888±387    48152±2365   125507±27686   ← +7.1% vs uniform
  bamor_nc     135992±17072   19786±2669   42678±6514   124706±24850
  bamor_ns     115195±39590   21645±648    36032±8253   109489±45003
  uniform      122258±54778   21069±1255   37567±18337  119411±55887
  fixed        117239±39450   14260±6239   34122±5586   102358±40090
```
BAMOR beats uniform on all modes, with 3× lower variance.
