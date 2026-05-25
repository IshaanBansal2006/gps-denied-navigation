#!/usr/bin/env bash
# Full desktop eval queue. Sequential (one GPU). Survives WSL idle via setsid+nohup.
#
# Queue (in order):
#   1. Cross-sequence eval (vanilla / RLS / continuous on 4 sequences)
#   2. Monte Carlo eval (20 random outage windows on MH_05)
#   3. v18 nav eval (vanilla / RLS / continuous on MH_05 at 5/10/30/60s)
#   4. Multi-duration eval on v15 (7 durations × 3 systems on MH_05)
#   5. Multi-duration eval on v18 (same)

set -o pipefail
cd /home/ishaan/projects/gps-denied-navigation
source .venv/bin/activate
mkdir -p logs

run() {
  local label="$1"; shift
  echo "[$(date -Is)] starting $label"
  "$@" 2>&1 | tee "logs/${label}.log"
  local rc=${PIPESTATUS[0]}
  echo "[$(date -Is)] $label exit=$rc"
  return "$rc"
}

run cross_seq         python3 -u scripts/cross_sequence_eval.py
run monte_carlo       python3 -u scripts/monte_carlo_eval.py
run v18_nav           python3 -u scripts/neural_aided_ekf_v18.py --outages 5,10,30,60
run multi_duration_v15 python3 -u scripts/multi_duration_eval.py --checkpoint checkpoints/lstm_v15.pt
run multi_duration_v18 python3 -u scripts/multi_duration_eval.py --checkpoint checkpoints/lstm_v18.pt

echo "[$(date -Is)] queue done"
exit 0
