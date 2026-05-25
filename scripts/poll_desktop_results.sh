#!/usr/bin/env bash
# Poll the desktop for completed training runs and rsync results back to the laptop.
# Idempotent: drops a .pulled marker so we don't re-pull.
# Designed for crontab use — logs to logs/poll_desktop.log, never writes to stderr on success.

set -u
set -o pipefail

REPO=/home/ishaan/projects/gps-denied-navigation
LOG=$REPO/logs/poll_desktop.log
SSH_HOST=desktop
REMOTE_REPO=/home/ishaan/projects/gps-denied-navigation

mkdir -p "$REPO/logs"

log() { echo "[$(date -Is)] $*" >> "$LOG"; }

# Bail quietly if SSH isn't reachable (laptop offline, desktop down, etc.).
# SSH lands in Windows PowerShell — use `exit 0` (works in both PowerShell and bash).
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$SSH_HOST" "exit 0" 2>/dev/null; then
  log "ssh unreachable, skipping"
  exit 0
fi

pull_run() {
  local run="$1"   # e.g. lstm_v15
  local marker="$REPO/results/$run/.pulled"
  if [ -f "$marker" ]; then
    return 0
  fi

  # Is the run finished on the desktop? Finished == test_metrics.json exists.
  if ! ssh "$SSH_HOST" "wsl -e bash -lc \"test -f $REMOTE_REPO/results/$run/test_metrics.json\"" 2>/dev/null; then
    log "$run: not done yet"
    return 0
  fi

  log "$run: complete, pulling results + checkpoint"

  # SSH lands in PowerShell on the desktop — `rsync` isn't a cmdlet there.
  # `--rsync-path="wsl rsync"` makes the remote side invoke rsync inside WSL.
  local RSYNC_OPTS=(-az --rsync-path="wsl rsync")

  # Pull results dir (small JSONs).
  if rsync "${RSYNC_OPTS[@]}" --info=stats1 \
      "$SSH_HOST:$REMOTE_REPO/results/$run/" "$REPO/results/$run/" >> "$LOG" 2>&1; then
    log "$run: results synced"
  else
    log "$run: results rsync FAILED"
    return 1
  fi

  # Pull checkpoint.
  if rsync "${RSYNC_OPTS[@]}" --info=stats1 \
      "$SSH_HOST:$REMOTE_REPO/checkpoints/${run}.pt" "$REPO/checkpoints/${run}.pt" >> "$LOG" 2>&1; then
    log "$run: checkpoint synced"
  else
    log "$run: checkpoint rsync FAILED (continuing — may not exist yet)"
  fi

  # Pull training log too (handy for debugging).
  rsync "${RSYNC_OPTS[@]}" "$SSH_HOST:$REMOTE_REPO/logs/${run}.log" "$REPO/logs/${run}.log" >> "$LOG" 2>&1 || true

  touch "$marker"
  log "$run: marked .pulled"
}

pull_run lstm_v15
pull_run lstm_v16
