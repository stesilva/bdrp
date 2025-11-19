#!/bin/bash

#SBATCH --job-name=rgcn_ppi5k_modes
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1

# Optional: Specify account if required
# #SBATCH --account=YOUR_PROJECT

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

# Load modules
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
source activate torch-env || {
    echo "ERROR: Failed to activate 'torch-env' conda environment"
    echo "Available environments:"
    conda env list
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
echo "Conda env:        $(conda info --envs | grep '*' | awk '{print $1}')"
echo ""

# Check and install dependencies
echo "Checking and installing dependencies..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || {
    echo "ERROR: PyTorch not found. Please install PyTorch first."
    exit 1
}

# Get CUDA version for torch-scatter installation
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda.replace('.', ''))" 2>/dev/null || echo "113")
echo "Detected CUDA version for torch-scatter: ${CUDA_VERSION}"

# Install torch-scatter if not available
python -c "import torch_scatter" 2>/dev/null || {
    echo "Installing torch-scatter..."
    # Try pip install first
    pip install torch-scatter -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)").html 2>/dev/null || \
    pip install torch-scatter 2>/dev/null || {
        echo "Warning: Could not install torch-scatter via pip"
        echo "You may need to install it manually:"
        echo "  pip install torch-scatter -f https://data.pyg.org/whl/torch-<version>.html"
    }
}

# Install torch-geometric if not available
python -c "import torch_geometric" 2>/dev/null || {
    echo "Installing torch-geometric..."
    pip install torch-geometric 2>/dev/null || {
        echo "Warning: Could not install torch-geometric via pip"
    }
}

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
    print(f'GPU memory:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
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
from torch_geometric.data import Data
print('PyG modules:      OK')
" || {
    echo "ERROR: PyTorch Geometric not installed or not working correctly"
    exit 1
}
echo ""

# Check torch-scatter
echo "Checking torch-scatter..."
python -c "
from torch_scatter import scatter_add
print('torch-scatter:     OK')
" || {
    echo "ERROR: torch-scatter not installed or not working correctly"
    echo "Try installing manually:"
    echo "  pip install torch-scatter -f https://data.pyg.org/whl/torch-\$(python -c 'import torch; print(torch.__version__)').html"
    exit 1
}
echo ""

# Check other dependencies
echo "Checking other dependencies..."
python -c "
import numpy as np
from tqdm import tqdm
print(f'NumPy version:    {np.__version__}')
print(f'tqdm:             OK')
" || {
    echo "ERROR: Required packages (numpy, tqdm) not found"
    exit 1
}
echo ""

# Check for required Python files
echo "Checking for required files..."
REQUIRED_FILES=("main.py" "models_edge_weight.py" "utils.py")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$SLURM_SUBMIT_DIR/$file" ]; then
        echo "ERROR: $file not found in $SLURM_SUBMIT_DIR"
        echo "Available Python files:"
        ls -la "$SLURM_SUBMIT_DIR"/*.py 2>/dev/null || echo "No Python files found"
        exit 1
    fi
    echo "Found: $file"
done
echo ""

# Check for dataset
echo "Checking for PPI5K dataset..."
if [ -d "$SLURM_SUBMIT_DIR/data/ppi5k" ]; then
    echo "Found: data/ppi5k directory"
    echo "Dataset files:"
    ls -lh "$SLURM_SUBMIT_DIR/data/ppi5k/"*.tsv "$SLURM_SUBMIT_DIR/data/ppi5k/"*.csv 2>/dev/null | head -5 || echo "Warning: Expected files not found"
    echo ""
    echo "Required files (ppi5k format):"
    echo "  - train.tsv"
    echo "  - val.tsv"
    echo "  - test.tsv"
elif [ -d "$SLURM_SUBMIT_DIR/data/ppi5k" ]; then
    echo "Found: data/ppi5k directory"
    echo "Dataset files:"
    ls -lh "$SLURM_SUBMIT_DIR/data/ppi5k/"*.tsv "$SLURM_SUBMIT_DIR/data/ppi5k/"*.csv 2>/dev/null | head -5 || echo "Warning: Expected files not found"
    echo ""
    echo "Required files (ppi5k format):"
    echo "  - train.tsv"
    echo "  - val.tsv"
    echo "  - test.tsv"
else
    echo "ERROR: data/ppi5k or data/ppi5k directory not found"
    echo "Please ensure the dataset is in: $SLURM_SUBMIT_DIR/data/ppi5k/"
    echo "Expected structure (ppi5k format):"
    echo "  data/ppi5k/train.tsv"
    echo "  data/ppi5k/val.tsv"
    echo "  data/ppi5k/test.tsv"
    exit 1
fi
echo ""

echo "======================================================================"
echo "Starting RGCN Training"
echo "======================================================================"
echo "Working directory: $(pwd)"
echo "Script:            main.py"
echo "GPU:               Enabled (device 0)"
echo "Test graph size:   All triplets (default for ppi5k, use --test-graph-size to change)"
echo "Output will be saved to: ${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
echo ""
echo "Note: If you get OOM errors, reduce --test-graph-size (e.g., 150000)"
echo "      Default uses all triplets for best results"
echo ""

# Set environment variables for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Set PyTorch CUDA memory allocator to reduce fragmentation
# This helps avoid OOM errors by allowing memory segments to expand
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run with error handling
cd "$SLURM_SUBMIT_DIR"

# Time the execution
START_TIME=$(date +%s)

# Run the training script with GPU enabled
# Default uses all triplets for ppi5k (smaller dataset, should fit in A100 memory)
# Adjust --test-graph-size if needed:
#   - 150000: safer, uses less memory
#   - 200000: balanced option
#   - -1: use all triplets (default for ppi5k)
python main.py --gpu 0 --test-graph-size -1 2>&1 || {
    EXIT_STATUS=$?
    echo ""
    echo "======================================================================"
    echo "ERROR: Python script failed with exit code $EXIT_STATUS"
    echo "======================================================================"
    echo ""
    echo "Troubleshooting tips:"
    echo "1. Check the error log: ${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
    echo "2. Verify all dependencies are installed"
    echo "3. Check if dataset files are present and readable"
    echo "4. Try running interactively: srun --pty bash"
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
echo "Model saved to:   best_mrr_model.pth"
echo ""
echo "For detailed resource usage, run:"
echo "  seff $SLURM_JOB_ID"
echo "======================================================================"

exit 0
