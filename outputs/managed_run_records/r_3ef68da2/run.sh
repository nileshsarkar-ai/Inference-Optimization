#!/bin/sh
set -u

LOG_FILE=/home/jl-runs/r_3ef68da2/output.log
PID_FILE=/home/jl-runs/r_3ef68da2/pid
EXIT_FILE=/home/jl-runs/r_3ef68da2/exit_code
COMMAND='if [ -x "$HOME/.venv/bin/python" ]; then export PATH="$HOME/.venv/bin:$PATH"; fi && bash -lc '"'"'/home/SaturateLLM/.venv/bin/python -u /home/SaturateLLM/graph_size_pilot/orchestrate.py'"'"''
WORKDIR='~'

rm -f "$EXIT_FILE"
touch "$LOG_FILE"
export PYTHONUNBUFFERED=1

child_pid=""
cleanup() {
  if [ -n "$child_pid" ]; then
    pgid=$(ps -o pgid= -p "$child_pid" 2>/dev/null | tr -d ' ')
    if [ -n "$pgid" ] && [ "$pgid" != "0" ]; then
      kill -- -"$pgid" 2>/dev/null || true
    else
      kill "$child_pid" 2>/dev/null || true
    fi
  fi
}

trap cleanup INT TERM

sh -lc "cd $WORKDIR && $COMMAND" >>"$LOG_FILE" 2>&1 &
child_pid=$!
printf '%s\n' "$child_pid" >"$PID_FILE"
wait "$child_pid"
status=$?
printf '%s\n' "$status" >"$EXIT_FILE"
exit 0
