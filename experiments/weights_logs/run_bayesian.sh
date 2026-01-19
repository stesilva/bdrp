#!/bin/bash

#SBATCH --job-name=rgcn_biokg_bayesian
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
    echo "Note: PyTorch 2.6.0+cu124 requires CUDA 12.4 compatible driver"
    echo "      If CUDA is not detected, try: module avail cuda"
    
    CUDA_LOADED=false
    for cuda_version in "12.4" "12.3" "12.2" "12.1" "12.0" "11.8" "11.7" "11.6" "11.5" "11.4" "11.3.1" "11.1.1"; do
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

# Install torch-scatter if not available
python -c "import torch_scatter" 2>/dev/null || {
    echo "Installing torch-scatter..."
    pip install torch-scatter -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)").html 2>/dev/null || \
    pip install torch-scatter 2>/dev/null || {
        echo "Warning: Could not install torch-scatter via pip"
    }
}

# Install torch-geometric if not available
python -c "import torch_geometric" 2>/dev/null || {
    echo "Installing torch-geometric..."
    pip install torch-geometric 2>/dev/null
}

# Install matplotlib and seaborn for analysis (if not already installed)
python -c "import matplotlib, seaborn" 2>/dev/null || {
    echo "Installing matplotlib and seaborn for edge weight analysis..."
    pip install matplotlib seaborn 2>/dev/null || echo "Warning: Could not install plotting libraries"
}

echo ""

# Check CUDA and PyTorch
echo "Checking CUDA driver and libraries..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA driver info:"
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null || echo "  nvidia-smi failed"
fi
echo ""

echo "Checking PyTorch..."
python -c "
import torch
print(f'PyTorch version:  {torch.__version__}')
print(f'CUDA available:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version:     {torch.version.cuda}')
    print(f'GPU device:       {torch.cuda.get_device_name(0)}')
    print(f'GPU memory:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
" || {
    echo "ERROR: PyTorch not installed or not working correctly"
    exit 1
}
echo ""

# Check for required Python files
echo "Checking for required files..."
REQUIRED_FILES=("main_bayesian.py" "models_edge_weight_bayesian.py" "utils.py")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$SLURM_SUBMIT_DIR/$file" ]; then
        echo "ERROR: $file not found in $SLURM_SUBMIT_DIR"
        exit 1
    fi
    echo "Found: $file"
done
echo ""

# Check for dataset
echo "Checking for biokg dataset..."
if [ -d "$SLURM_SUBMIT_DIR/data/biokg" ]; then
    echo "Found: data/biokg directory"
    ls -lh "$SLURM_SUBMIT_DIR/data/biokg/"*.tsv 2>/dev/null | head -5 || echo "Warning: Expected .tsv files not found"
else
    echo "ERROR: data/biokg directory not found"
    exit 1
fi
echo ""

# ============================================================================
# EDGE WEIGHT LOGGING CONFIGURATION
# ============================================================================
# Set these variables to control logging behavior
ENABLE_LOGGING=true              # Set to false to disable logging
LOG_FREQUENCY=500                # Log every N epochs (500 = every evaluation)
LOG_VALIDATION=false             # Also log during validation (increases storage)
EDGE_WEIGHT_MODE="learnable"     # "learnable" or "bayesian"

# Use simple log directory (will be created by the Python script)
LOG_DIR="edge_weight_logs"

echo "======================================================================"
echo "Edge Weight Logging Configuration"
echo "======================================================================"
echo "Logging enabled:     $ENABLE_LOGGING"
echo "Edge weight mode:    $EDGE_WEIGHT_MODE"
echo "Log frequency:       Every $LOG_FREQUENCY epochs"
echo "Log validation:      $LOG_VALIDATION"
echo "Log directory:       $LOG_DIR"
echo "======================================================================"
echo ""

# ============================================================================
# BUILD COMMAND LINE ARGUMENTS
# ============================================================================
CMD_ARGS=(
    --gpu 0
    --edge-weight-mode "$EDGE_WEIGHT_MODE"
    --test-graph-size -1
    --n-epochs 10000
    --evaluate-every 500
    --graph-batch-size 30000
    --n-bases 4
    --dropout 0.2
    --lr 0.01
    --regularization 0.01
    --seed 42
)

# Add logging arguments if enabled
if [ "$ENABLE_LOGGING" = true ]; then
    CMD_ARGS+=(
        --log-edge-weights
        --log-dir "$LOG_DIR"
        --log-frequency "$LOG_FREQUENCY"
    )
    if [ "$LOG_VALIDATION" = true ]; then
        CMD_ARGS+=(--log-validation)
    fi
fi

echo "======================================================================"
echo "Starting RGCN Training with Edge Weight Analysis"
echo "======================================================================"
echo "Working directory: $(pwd)"
echo "Script:            main_bayesian.py"
echo "Edge weight mode:  $EDGE_WEIGHT_MODE"
echo ""
echo "Command line arguments:"
for arg in "${CMD_ARGS[@]}"; do
    echo "  $arg"
done
echo ""
echo "Output will be saved to: ${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
echo "======================================================================"
echo ""

# Set environment variables for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run with error handling
cd "$SLURM_SUBMIT_DIR"

# Time the execution
START_TIME=$(date +%s)

# Run the training script
python main_bayesian.py "${CMD_ARGS[@]}" 2>&1 || {
    EXIT_STATUS=$?
    echo ""
    echo "======================================================================"
    echo "ERROR: Python script failed with exit code $EXIT_STATUS"
    echo "======================================================================"
    echo ""
    echo "Check the error log: ${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
    exit $EXIT_STATUS
}

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# ============================================================================
# POST-PROCESSING: ANALYZE EDGE WEIGHTS
# ============================================================================
if [ "$ENABLE_LOGGING" = true ] && [ -d "$LOG_DIR" ]; then
    echo ""
    echo "======================================================================"
    echo "Running Edge Weight Analysis"
    echo "======================================================================"
    
    # Check if analysis script exists
    if [ -f "analyze_bayesian_weights.py" ]; then
        echo "Generating visualizations and analysis report..."
        python analyze_bayesian_weights.py --log-dir "$LOG_DIR" 2>&1 || {
            echo "Warning: Analysis script failed, but training completed successfully"
        }
        
        echo ""
        echo "Analysis complete! Results saved to:"
        echo "  $LOG_DIR/"
        echo ""
        echo "Generated files:"
        ls -lh "$LOG_DIR"/*.png 2>/dev/null | awk '{print "  " $9}' || echo "  No plots generated"
        echo ""
    else
        echo "Note: analyze_bayesian_weights.py not found - skipping analysis"
        echo "You can analyze the logs later using:"
        echo "  python analyze_bayesian_weights.py --log-dir $LOG_DIR"
        echo ""
    fi
fi

# Success message
echo "======================================================================"
echo "Job Completed Successfully"
echo "======================================================================"
echo "Finished:         $(date)"
echo "Elapsed time:     $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
echo "Output file:      ${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
echo "Error file:       ${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
echo ""
echo "Model saved to:   best_mrr_model.pth"
if [ "$ENABLE_LOGGING" = true ]; then
    echo "Edge weight logs: $LOG_DIR/"
fi
echo ""
echo "For detailed resource usage, run:"
echo "  seff $SLURM_JOB_ID"
echo "======================================================================"

exit 0