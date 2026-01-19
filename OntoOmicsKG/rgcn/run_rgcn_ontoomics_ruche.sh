#!/bin/bash

#SBATCH --job-name=rgcn_ontoomics
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "======================================================================"
echo "Job Information"
echo "======================================================================"
echo "Job ID:           $SLURM_JOB_ID"
echo "Job name:         $SLURM_JOB_NAME"
echo "Node:             $SLURM_NODELIST"
echo "Partition:        $SLURM_JOB_PARTITION"
echo "CPUs:             $SLURM_CPUS_PER_TASK"
echo "Memory:           ${SLURM_MEM_PER_NODE}MB"
echo "Working dir:      $SLURM_SUBMIT_DIR"
echo "Started:          $(date)"
echo "======================================================================"

# Load modules
echo ""
echo "Loading modules..."
module purge
module load anaconda3/2020.02/gcc-9.2.0

# Load CUDA if using GPU
if [[ "$SLURM_JOB_PARTITION" == *"gpu"* ]]; then
    echo "Loading CUDA module..."
    module load cuda/11.3.1/gcc-9.2.0 2>/dev/null || \
    module load cuda/11.1.1/gcc-9.2.0 2>/dev/null || \
    module load cuda/10.2.89/intel-19.0.3.199 2>/dev/null || \
    echo "Warning: No CUDA module loaded. Using system CUDA if available."
fi

echo "Loaded modules:"
module list
echo ""

# Activate conda environment
echo "Activating conda environment..."
source activate torch-env || {
    echo "ERROR: Failed to activate 'torch-env' conda environment"
    echo "Available environments:"
    conda env list
    exit 1
}

echo "Environment activated: torch-env"
echo ""

# Navigate to the rgcn directory
SCRIPT_DIR="$SLURM_SUBMIT_DIR"
if [ ! -d "$SCRIPT_DIR" ]; then
    echo "ERROR: Directory not found: $SCRIPT_DIR"
    exit 1
fi

cd "$SCRIPT_DIR"
echo "Changed to directory: $(pwd)"
echo ""

# Check for required Python files
echo "Checking for required files..."
REQUIRED_FILES=("main.py" "models.py" "utils.py")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: $file not found in $SCRIPT_DIR"
        exit 1
    fi
    echo "Found: $file"
done
echo ""

# Check for OntoOmicsKG dataset (parent directory)
echo "Checking for OntoOmicsKG dataset..."
ONTOOMICS_PATH="$SLURM_SUBMIT_DIR/.."
if [ -d "$ONTOOMICS_PATH" ]; then
    echo "Found: OntoOmicsKG directory"
    if [ ! -f "$ONTOOMICS_PATH/edges.filtered.tsv.entities.tsv" ] || \
       [ ! -f "$ONTOOMICS_PATH/edges.filtered.tsv.relations.tsv" ] || \
       [ ! -f "$ONTOOMICS_PATH/edges_train.tsv" ] || \
       [ ! -f "$ONTOOMICS_PATH/edges_val.tsv" ] || \
       [ ! -f "$ONTOOMICS_PATH/edges_test.tsv" ]; then
        echo "ERROR: Some required OntoOmicsKG files are missing"
        exit 1
    fi
else
    echo "ERROR: OntoOmicsKG directory not found"
    exit 1
fi
echo ""

echo "======================================================================"
echo "Starting RGCN Training for OntoOmicsKG"
echo "======================================================================"
echo "Working directory: $(pwd)"
echo "Script:            main.py"
echo "Data path:         $ONTOOMICS_PATH"
echo "GPU:               Enabled (device 0)"
echo ""

# Set environment variables for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Time the execution
START_TIME=$(date +%s)

# Run the training script with GPU enabled
python main.py --gpu 0 --data-path "$ONTOOMICS_PATH" 2>&1 || {
    EXIT_STATUS=$?
    echo ""
    echo "======================================================================"
    echo "ERROR: Python script failed with exit code $EXIT_STATUS"
    echo "======================================================================"
    exit $EXIT_STATUS
}

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Success message
echo ""
echo "======================================================================"
echo "Job Completed Successfully"
echo "======================================================================"
echo "Finished:         $(date)"
echo "Elapsed time:     $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
echo "Model saved to:   $SCRIPT_DIR/best_mrr_model.pth"
echo "======================================================================"

exit 0

