#!/usr/bin/env bash
# Usage: slurm/submit.sh NGPUS CONFIG MODEL1 TP1 [MODEL2 TP2]   (NGPUS must equal TP1 + TP2)
# Reads partition/QoS/log dir from serve/site.env. Prints the sbatch command and asks before submitting.
set -euo pipefail
source serve/site.env
NG=$1; shift
WANT=$(( $3 + ${5:-0} )); [[ $NG -eq $WANT ]] || { echo "NGPUS=$NG but TP1+TP2=$WANT" >&2; exit 1; }
QARG=(); [[ -n "${QOS:-}" ]] && QARG=(-q "$QOS")
CMD=(sbatch -p "$PARTITION" "${QARG[@]}" --gres=gpu:"$NG" -o "$LOGDIR/%x_%j.out" slurm/run_stage.sbatch "$@")
sbatch --test-only -p "$PARTITION" "${QARG[@]}" --gres=gpu:"$NG" --time=24:00:00 --wrap=true 2>&1 | tail -1
echo "Will run: ${CMD[*]}"; read -r -p "Submit? [y/N] " ok; [[ $ok == y ]] && "${CMD[@]}"
