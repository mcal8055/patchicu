#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH --partition=<YOUR_CPU_PARTITION>
#SBATCH --qos=<YOUR_QOS>
#SBATCH --account=<YOUR_CPU_ACCOUNT>
#SBATCH --requeue
#SBATCH -J wandb_agent_cpu
#SBATCH -o ${SCRATCH}/patchicu/wandb_agent_cpu_%j.out
#SBATCH -e ${SCRATCH}/patchicu/wandb_agent_cpu_%j.err

# Wandb agent for CPU-only sweeps (LogisticRegression, LGBM, etc.) on notchpeak.
# Originally written for the University of Utah CHPC cluster; adjust the
# SLURM directives above (account, partition, qos) for your environment.
#
# Submit as: sbatch chpc_notchpeak_cpu_agent.sh <sweep_id>

SWEEP_ID=${1:?"Usage: sbatch chpc_notchpeak_cpu_agent.sh <sweep_id>"}

echo "=== CPU agent starting at $(date) on $(hostname) for sweep: $SWEEP_ID ==="

# --- Notchpeak activation gotcha: conda's yaib_env auto-loads, blocks venv ---
# Hence: conda deactivate FIRST, then explicit PATH export
conda deactivate 2>/dev/null || true
export PATH=${SCRATCH}/patchicu/yaib_env_x86/bin:$PATH

# --- FD + thread fixes (same as granite agent) ---
ulimit -n 65536 2>/dev/null || ulimit -n 4096
export OMP_NUM_THREADS=1

echo "=== Python: $(which python3) ==="
python3 --version
python3 -c "import wandb, icu_benchmarks; print('wandb:', wandb.__version__, '| YAIB importable')"

# --- Launch wandb agent ---
# Use `python3 -m wandb` form — the `wandb` binary shebang in yaib_env_x86 points
# at a different venv path (yaib_env) which may not exist. `python3 -m` avoids this.
cd ${SCRATCH}/patchicu/YAIB
echo "=== Starting wandb agent at $(date) ==="
python3 -m wandb agent --count 1 "$SWEEP_ID"

echo "=== CPU agent finished at $(date) ==="
