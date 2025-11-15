#!/bin/bash
#SBATCH --job-name=rgcn_nl27k
#SBATCH --output=rgcn_nl27k_%j.out
#SBATCH --error=rgcn_nl27k_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo "======================================================================"
echo "Job started on $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo "======================================================================"

# Load necessary modules (adjust based on what's available on Ruche)
# Check available modules with: module avail python
module purge
module load python/3.9.10/gcc-11.2.0  # Use Python 3.6+ for f-string support

# Activate virtual environment
# Make sure you've created it in /workdir/silvas/bdrp/venv
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
echo "Key packages:"
python -c "import torch; import numpy; print(f'PyTorch: {torch.__version__}'); print(f'NumPy: {numpy.__version__}')"
echo ""

# Change to the directory where your script is located
cd $WORKDIR/bdrp

# Run the prediction script
echo "Starting training..."
python predict_links.py

echo ""
echo "Job completed at $(date)"

