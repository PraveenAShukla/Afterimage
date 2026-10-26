#!/usr/bin/env bash
#SBATCH --job-name=x0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=03:00:00
# X0 (docs/AFTERIMAGE.md): all-mock afterimage positive control on one CPU node, no GPU, no API. Runs every
# configs/afterimage/x0 config (8 at a time; each runs its 24 seeds in parallel), then x0_mc.py and afterimage.py.
# Runs skip when complete, so resubmitting resumes.  Usage: sbatch -p <cpu partition> slurm/x0_cpu.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source serve/site.env; set +a
source "${VENV_OVERRIDE:-$VENV}/bin/activate"
export RUNS_DIR=${RUNS_DIR:-runs_afterimage_x0}
scripts/make_runs_dir.sh "$RUNS_DIR" runs
echo "[$(date -u)] X0 runs -> $RUNS_DIR"
ls configs/afterimage/x0/*.yaml | xargs -P 8 -I{} sh -c 'python -m collusion.run "$1" --runs-dir "$RUNS_DIR" > /dev/null 2>&1 || echo "FAILED $1"' _ {}
echo "[$(date -u)] runs done: $(ls -d "$RUNS_DIR"/x0* | wc -l) run dirs"
python analysis/x0_mc.py --runs-dir "$RUNS_DIR"
python analysis/afterimage.py --runs-dir runs --runs-dir "$RUNS_DIR"
echo "[$(date -u)] X0 done"
