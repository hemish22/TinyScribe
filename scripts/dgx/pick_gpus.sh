#!/usr/bin/env bash
# Pick free GPUs on the shared DGX without going over this account's GPU budget.
#   GPUS=$(bash scripts/dgx/pick_gpus.sh 3)      # up to 3 free GPUs, e.g. "1,6,7"; empty if none
# The account (all of $USER's processes, including other people's work under the same login) may
# hold at most MAX_GPUS GPUs (default 4). A GPU is free when it has no compute process and under
# 1 GB in use. A GPU claimed by wait_gpus.sh in the last CLAIM_SECS seconds (runs/gpu_claims/<i>)
# counts as ours and busy, since a job takes a minute to load its model. Prints a summary to stderr.
set -eo pipefail   # no -u: empty associative arrays count as unset in bash 4
WANT="${1:-1}"; MAX_GPUS="${MAX_GPUS:-4}"; ME="${USER:-$(id -un)}"
CLAIMS="$(cd "$(dirname "$0")/../.." && pwd)/runs/gpu_claims"; CLAIM_SECS="${CLAIM_SECS:-180}"

declare -A idx_of busy mine
while IFS=, read -r i uuid mem; do
  i=${i// /}; uuid=${uuid// /}; mem=${mem// /}
  idx_of[$uuid]=$i
  [ "${mem%% *}" -ge 1000 ] && busy[$i]=1
done < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)

while IFS=, read -r uuid pid; do
  uuid=${uuid// /}; pid=${pid// /}; [ -n "$pid" ] || continue
  i=${idx_of[$uuid]:-}; [ -n "$i" ] || continue
  busy[$i]=1
  [ "$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')" = "$ME" ] && mine[$i]=1
done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader)

now=$(date +%s)
for f in "$CLAIMS"/*; do
  [ -f "$f" ] || continue
  [ $((now - $(stat -c %Y "$f"))) -lt "$CLAIM_SECS" ] && { mine[$(basename "$f")]=1; busy[$(basename "$f")]=1; }
done
used=${#mine[@]}
budget=$((MAX_GPUS - used)); [ "$budget" -lt 0 ] && budget=0
n=$WANT; [ "$n" -gt "$budget" ] && n=$budget
picked=()
for i in $(printf '%s\n' "${idx_of[@]}" | sort -n); do
  [ "${#picked[@]}" -ge "$n" ] && break
  [ -z "${busy[$i]:-}" ] && picked+=("$i")
done
echo "$ME holds ${used} GPU(s) [${!mine[*]}]; budget ${budget}; picked ${picked[*]:-none}" >&2
(IFS=,; echo "${picked[*]:-}")
