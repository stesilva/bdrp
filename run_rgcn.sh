#!/bin/bash
#SBATCH --job-name=rgcn_cn15k
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1


# Optional: Email notifications
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=your.email@centralesupelec.fr

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

# Load modules for Ruche
echo ""
echo "Loading modules..."
module purge
module load anaconda3/2020.02/gcc-9.2.0

# Load CUDA if using GPU
if [[ "$SLURM_JOB_PARTITION" == *"gpu"* ]]; then
    echo "Loading CUDA module..."
    # Check available CUDA versions: module avail cuda
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
source activate torch-env || \
conda activate torch-env || {
    echo "ERROR: Failed to activate 'torch-env' conda environment"
    echo "Available environments:"
    conda env list
    echo ""
    echo "To create the environment, run:"
    echo "  conda create -n torch-env python=3.9"
    echo "  conda activate torch-env"
    echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    echo "  pip install torch-geometric"
    echo "  pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html"
    exit 1
}

echo "Environment activated: torch-env"
echo ""

# Print environment information
echo "======================================================================"
echo "Environment Information"
echo "======================================================================"
echo "Python:           $(python --version 2>&1)"
echo "Python path:      $(which python)"
echo "Conda env:        $CONDA_DEFAULT_ENV"
echo ""

# Check PyTorch installation
echo "Checking PyTorch..."
python -c "
import torch
print(f'PyTorch version:  {torch.__version__}')
print(f'CUDA available:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version:     {torch.version.cuda}')
    print(f'GPU device:       {torch.cuda.get_device_name(0)}')
    print(f'GPU count:        {torch.cuda.device_count()}')
else:
    print('WARNING: CUDA not available - will use CPU only')
" || {
    echo "ERROR: PyTorch not installed or not working correctly"
    exit 1
}
echo ""

# Check PyTorch Geometric
echo "Checking PyTorch Geometric..."
python -c "
import torch_geometric
print(f'PyG version:      {torch_geometric.__version__}')
from torch_geometric.nn import RGCNConv
print('PyG modules:      OK')
" || {
    echo "ERROR: PyTorch Geometric not installed or not working correctly"
    exit 1
}
echo ""

# Check other dependencies
echo "Checking other dependencies..."
python -c "
import numpy as np
import sklearn
print(f'NumPy version:    {np.__version__}')
print(f'scikit-learn:     {sklearn.__version__}')
" || {
    echo "ERROR: Required packages (numpy, scikit-learn) not found"
    exit 1
}
echo ""

# Check for required files
echo "Checking for required files..."
REQUIRED_FILES=("predict_links.py" "models.py" "layers.py" "rgcn_conv.py" "rgcn_utils.py")
MISSING_FILES=0

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$SLURM_SUBMIT_DIR/$file" ]; then
        echo "Found: $file"
    else
        echo "ERROR: Missing $file"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

if [ $MISSING_FILES -gt 0 ]; then
    echo ""
    echo "ERROR: $MISSING_FILES required file(s) missing"
    exit 1
fi
echo ""

# Check for utils directory
if [ -d "$SLURM_SUBMIT_DIR/utils" ]; then
    echo "Found: utils/ directory"
    ls -la "$SLURM_SUBMIT_DIR/utils/"*.py 2>/dev/null || echo "Warning: No Python files in utils/"
else
    echo "ERROR: utils/ directory not found"
    exit 1
fi
echo ""

# Check for dataset
echo "Checking for CN15K dataset..."
if [ -d "$SLURM_SUBMIT_DIR/cn15k" ]; then
    echo "Found: cn15k/ directory"
    echo "Dataset files:"
    ls -lh "$SLURM_SUBMIT_DIR/cn15k/"*.tsv 2>/dev/null || echo "Warning: No .tsv files found"
else
    echo "ERROR: cn15k/ directory not found"
    echo "Please ensure the dataset is in: $SLURM_SUBMIT_DIR/cn15k/"
    exit 1
fi
echo ""

echo "======================================================================"
echo "Starting R-GCN Link Prediction Training"
echo "======================================================================"
echo "Working directory: $(pwd)"
echo "Script:            predict_links.py"
echo "Output will be saved to: ${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
echo ""

# Set environment variables for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Prevent PyTorch from using too many threads
export OMP_WAIT_POLICY=PASSIVE

# Run with error handling
cd "$SLURM_SUBMIT_DIR"

# Time the execution
START_TIME=$(date +%s)

python predict_links.py 2>&1 || {
    EXIT_STATUS=$?
    echo ""
    echo "======================================================================"
    echo "ERROR: Python script failed with exit code $EXIT_STATUS"
    echo "======================================================================"
    echo ""
    echo "Troubleshooting tips:"
    echo "1. Check the error log: ${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
    echo "2. Verify all dependencies are installed in torch-env"
    echo "3. Check if dataset files are present and readable"
    echo "4. Try running interactively: srun --partition=gpu_prod_long --gres=gpu:1 --pty bash"
    echo "5. Check available partitions: sinfo"
    echo ""
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
echo "Output file:      ${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
echo "Error file:       ${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
echo ""
echo "For detailed resource usage, run:"
echo "  seff $SLURM_JOB_ID"
echo "======================================================================"

exit 0