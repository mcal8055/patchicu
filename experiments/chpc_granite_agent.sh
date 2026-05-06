#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=198G
#SBATCH -t 24:00:00
#SBATCH --partition=<YOUR_GPU_PARTITION>
#SBATCH --qos=<YOUR_QOS>
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --requeue
#SBATCH --gres=gpu:gh200:1
#SBATCH -J wandb_agent
#SBATCH -o ${SCRATCH}/patchicu/wandb_agent_%j.out
#SBATCH -e ${SCRATCH}/patchicu/wandb_agent_%j.err

# Wandb agent for the PatchICU HiRID sepsis 1hr sweep on granite GH200.
# Originally written for the University of Utah CHPC cluster; adjust the
# SLURM directives above (account, partition, qos) for your environment.
#
# Submit as: sbatch chpc_granite_agent.sh <sweep_id>
# where <sweep_id> is the full path like "verity-hogans-university-of-utah/patchicu/abc12345"
# (printed by `wandb sweep <yaml>`).
#
# The agent pulls trials from wandb's central coordinator and runs them sequentially.
# With --requeue, SLURM will resubmit after preemption / walltime, and the agent
# picks up wherever wandb's state left off.

SWEEP_ID=${1:?"Usage: sbatch chpc_granite_agent.sh <sweep_id>"}

echo "=== Agent starting at $(date) on $(hostname) for sweep: $SWEEP_ID ==="

# --- Environment fixes applied in earlier iterations ---
ulimit -n 65536 2>/dev/null || ulimit -n 4096      # prevent FD exhaustion
export OMP_NUM_THREADS=1                            # avoid worker thread contention
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # HBM fragmentation

# --- Load Python 3.12 module (required for rebuilt venv) ---
module load python/3.12.4

# --- Activate venv ---
cd ${SCRATCH}/patchicu
source yaib_env_arm/bin/activate

echo "=== Python: $(which python3) ==="
python3 -c "import wandb, icu_benchmarks; print('wandb:', wandb.__version__, '| YAIB importable')"

# --- Run the wandb agent ---
# The agent runs in YAIB's working dir so relative paths in the sweep YAML (e.g. configs/)
# resolve correctly.
cd YAIB
echo "=== Starting wandb agent at $(date) ==="
wandb agent --count 5 "$SWEEP_ID"

echo "=== Agent finished at $(date) ==="
