#!/usr/bin/env bash
# Wait until at least MIN free GPUs are available within the account's budget (pick_gpus.sh), then
# print up to WANT of them (e.g. "6,7") and claim them for a few minutes so a second waiting job
# cannot take the same GPUs before this one has loaded its model.
#   GPUS=$(bash scripts/dgx/wait_gpus.sh 4 1)      # up to 4, at least 1
set -eo pipefail
cd "$(dirname "$0")/../.."
WANT="${1:-1}"; MIN="${2:-$WANT}"; POLL="${POLL:-120}"
mkdir -p runs/gpu_claims
exec 9>runs/gpu_claims.lock
last=""
while true; do
  flock 9
  g=$(bash scripts/dgx/pick_gpus.sh "$WANT" 2>/tmp/ts_pick_$$ || true)
  n=$(echo "$g" | tr ',' '\n' | grep -c . || true)
  if [ "$n" -ge "$MIN" ]; then
    for i in ${g//,/ }; do touch "runs/gpu_claims/$i"; done
    flock -u 9; rm -f /tmp/ts_pick_$$
    echo "$g"; exit 0
  fi
  flock -u 9
  msg=$(cat /tmp/ts_pick_$$); [ "$msg" != "$last" ] && echo "$(date '+%F %T') waiting: $msg" >&2; last="$msg"
  sleep "$POLL"
done
