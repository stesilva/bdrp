#!/bin/bash
#SBATCH --job-name=reddit_bayesian
#SBATCH --output=reddit_bayesian_%j.out
#SBATCH --error=reddit_bayesian_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

echo "======================================================================"
echo "Reddit Bayesian RGCN Experiment"
echo "Job started on $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

# Load necessary modules
module purge
module load python/3.9.10/gcc-11.2.0
module load cuda/11.8.0/gcc-11.2.0

# Change to the experiment directory (use absolute path)
cd /workdir/silvas/bdrp/newdatasets/reddit
echo "Changed to directory: $(pwd)"
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Warning: venv not found, using system Python"
fi

# Verify Python and packages
echo ""
echo "Python version:"
python --version
echo ""
echo "CUDA availability:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Reddit data paths (converted data should already exist)
REDDIT_DATA_DIR="reddit_converted"

# Step 1: Check if converted data exists (conversion should be done locally)
if [ ! -f "$REDDIT_DATA_DIR/train.tsv" ]; then
    echo "ERROR: Converted Reddit data not found at $REDDIT_DATA_DIR/train.tsv"
    echo "Current directory: $(pwd)"
    echo "Contents:"
    ls -la
    echo ""
    echo "reddit_converted contents:"
    ls -la reddit_converted/ 2>/dev/null || echo "reddit_converted directory not found"
    exit 1
else
    echo "Found converted Reddit data at $REDDIT_DATA_DIR/train.tsv"
fi

# Step 2: Run Bayesian RGCN
echo ""
echo "======================================================================"
echo "Starting Bayesian RGCN training on Reddit dataset..."
echo "======================================================================"
echo ""

python3 main_bayesian.py \
    --edge-weight-mode bayesian \
    --gpu 0 \
    --lr 1e-2 \
    --n-epochs 10000 \
    --evaluate-every 500 \
    --graph-batch-size 50000 \
    --graph-split-size 0.5 \
    --negative-sample 1 \
    --dropout 0.2 \
    --n-bases 4 \
    --regularization 1e-2 \
    --grad-norm 1.0 \
    --test-graph-size 200000

echo ""
echo "======================================================================"
echo "Job completed at $(date)"
echo "======================================================================"
