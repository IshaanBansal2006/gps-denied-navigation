#!/usr/bin/env bash
# Desktop runner: cross-sequence eval, then Monte Carlo eval.
# Both run sequentially to avoid GPU contention.

set -o pipefail
cd /home/ishaan/projects/gps-denied-navigation
source .venv/bin/activate
mkdir -p logs

echo "[$(date -Is)] starting cross_sequence_eval.py"
python3 -u scripts/cross_sequence_eval.py 2>&1 | tee logs/cross_seq.log
RC1=${PIPESTATUS[0]}
echo "[$(date -Is)] cross_sequence exit=$RC1"

echo "[$(date -Is)] starting monte_carlo_eval.py"
python3 -u scripts/monte_carlo_eval.py 2>&1 | tee logs/monte_carlo.log
RC2=${PIPESTATUS[0]}
echo "[$(date -Is)] monte_carlo exit=$RC2"

echo "[$(date -Is)] all done  cross=$RC1  mc=$RC2"
exit 0
