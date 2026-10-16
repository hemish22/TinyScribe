#!/usr/bin/env bash
# Run a command in its own window of the tmux session "ts", so it survives SSH disconnects.
# Output goes to runs/logs/<name>.log and the exit code to runs/logs/<name>.exit when it ends.
#   bash scripts/dgx/tjob.sh setup bash scripts/dgx/setup_env.sh
#   tail -f runs/logs/setup.log         # follow;  tmux attach -t ts   # watch live
set -euo pipefail
cd "$(dirname "$0")/../.."
NAME="$1"; shift
mkdir -p runs/logs
rm -f "runs/logs/$NAME.exit"
# write the job as a script so arguments keep their quoting (labels with spaces, parentheses)
JOB="runs/logs/$NAME.job.sh"
{
  echo "cd $(printf %q "$PWD")"
  printf '%q ' "$@"; echo "> runs/logs/$NAME.log 2>&1"
  echo "echo \$? > runs/logs/$NAME.exit"
} > "$JOB"
tmux has-session -t ts 2>/dev/null || tmux new-session -d -s ts -n shell
tmux new-window -t ts -n "$NAME" "bash -l $PWD/$JOB"
echo "started '$NAME' in tmux session ts (log: runs/logs/$NAME.log)"
