#!/bin/bash

#SBATCH --job-name=rgcn_cn15k_modes
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
# Note: PyTorch 2.6.0+cu124 comes with bundled CUDA libraries, but we still need
# compatible CUDA driver and may need system CUDA for compilation tools
if [[ "$SLURM_JOB_PARTITION" == *"gpu"* ]]; then
    echo "Loading CUDA module..."
    echo "Note: PyTorch 2.6.0+cu124 requires CUDA 12.4 compatible driver"
    echo "      If CUDA is not detected, try: module avail cuda"
    
    # Try CUDA 12.x first (for PyTorch 2.6.0+cu124), then fall back to 11.x
    # Skip 10.x as it's incompatible with PyTorch 2.6.0+cu124
    CUDA_LOADED=false
    for cuda_version in "12.4" "12.3" "12.2" "12.1" "12.0" "11.8" "11.7" "11.6" "11.5" "11.4" "11.3.1" "11.1.1"; do
        # Try different compiler variants
        if module load cuda/${cuda_version}/gcc-9.2.0 2>/dev/null; then
            echo "Successfully loaded CUDA ${cuda_version} with gcc-9.2.0"
            CUDA_LOADED=true
            break
        elif module load cuda/${cuda_version}/intel-19.0.3.199 2>/dev/null; then
            echo "Successfully loaded CUDA ${cuda_version} with intel-19.0.3.199"
            CUDA_LOADED=true
            break
        fi
    done
    
    if [ "$CUDA_LOADED" = false ]; then
        echo "Warning: Could not load CUDA 11.x or 12.x module."
        echo "PyTorch 2.6.0+cu124 may still work with bundled CUDA libraries if driver is compatible."
        echo "Available CUDA modules (check with: module avail cuda):"
        module avail cuda 2>&1 | head -20 || true
        echo ""
        echo "If CUDA is still not detected, you may need to:"
        echo "  1. Check GPU driver version: nvidia-smi"
        echo "  2. Install PyTorch version matching available CUDA"
        echo "  3. Or use CPU mode: --gpu -1"
    fi
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

# Check CUDA driver and libraries
echo "Checking CUDA driver and libraries..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA driver info:"
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null || echo "  nvidia-smi failed"
else
    echo "  nvidia-smi not found in PATH"
fi
if [ -n "${CUDA_HOME:-}" ]; then
    echo "CUDA_HOME: $CUDA_HOME"
fi
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    echo "LD_LIBRARY_PATH contains CUDA: $(echo $LD_LIBRARY_PATH | grep -o 'cuda[^:]*' | head -1 || echo 'none')"
fi
echo ""

# Check PyTorch installation
echo "Checking PyTorch..."
python -c "
import torch
import os
print(f'PyTorch version:  {torch.__version__}')
print(f'PyTorch CUDA compiled version: {torch.version.cuda if hasattr(torch.version, \"cuda\") else \"N/A\"}')
print(f'CUDA available:   {torch.cuda.is_available()}')
if not torch.cuda.is_available():
    print('CUDA not available. Possible reasons:')
    print('  1. CUDA driver version too old')
    print('  2. CUDA library version mismatch')
    print('  3. GPU not accessible')
    print(f'  CUDA_HOME: {os.environ.get(\"CUDA_HOME\", \"not set\")}')
    print(f'  LD_LIBRARY_PATH: {os.environ.get(\"LD_LIBRARY_PATH\", \"not set\")[:200]}')
else:
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
echo "Checking for CN15K dataset..."
if [ -d "$SLURM_SUBMIT_DIR/data/cn15k" ]; then
    echo "Found: data/cn15k directory"
    echo "Dataset files:"
    ls -lh "$SLURM_SUBMIT_DIR/data/cn15k/"*.tsv "$SLURM_SUBMIT_DIR/data/cn15k/"*.csv 2>/dev/null | head -5 || echo "Warning: Expected files not found"
    echo ""
    echo "Required files (cn15k format):"
    echo "  - train.tsv"
    echo "  - val.tsv"
    echo "  - test.tsv"
else
    echo "ERROR: data/cn15k or data/cn15k directory not found"
    echo "Please ensure the dataset is in: $SLURM_SUBMIT_DIR/data/cn15k/"
    echo "Expected structure (cn15k format):"
    echo "  data/cn15k/train.tsv"
    echo "  data/cn15k/val.tsv"
    echo "  data/cn15k/test.tsv"
    exit 1
fi
echo ""

echo "======================================================================"
echo "Starting RGCN Training"
echo "======================================================================"
echo "Working directory: $(pwd)"
echo "Script:            main.py"
echo "GPU:               Enabled (device 0)"
echo "Test graph size:   All triplets (default for cn15k, use --test-graph-size to change)"
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
# Default uses all triplets for cn15k (smaller dataset, should fit in A100 memory)
# Adjust --test-graph-size if needed:
#   - 150000: safer, uses less memory
#   - 200000: balanced option
#   - -1: use all triplets (default for cn15k)
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
