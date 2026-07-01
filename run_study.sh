#!/usr/bin/env bash
# Run the full study end to end and write the tables and figures.
# Reason-step results are cached in results/cache/, so re-runs are fast.
#
#   MODEL=gpt-4o-mini ./run_study.sh   # default
#   MODEL=gpt-4o      ./run_study.sh   # stronger model for the final runs
set -e
cd "$(dirname "$0")/src"
PY="python3"
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
MODEL="${MODEL:-gpt-4o-mini}"; export MODEL
echo ">>> MODEL=$MODEL"

echo ">>> Reddit (POOL-EN): LOO + Additive + Interaction + STYLE"
$PY -u experiments.py all EN

echo ">>> Hacker News (POOL-HN): LOO + Additive (replication)"
$PY -u experiments.py loo HN
$PY -u experiments.py additive HN

echo ">>> Qualitative case dumps"
$PY -u qualitative.py en
$PY -u qualitative.py hn || true

echo ">>> Tables + figures"
$PY -u report.py
echo ">>> DONE. See results/tables/ and results/figures/"
