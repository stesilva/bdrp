#!/bin/bash
#SBATCH --job-name=rgcn_nl27k_gpu
#SBATCH --output=rgcn_nl27k_gpu_%j.out
#SBATCH --error=rgcn_nl27k_gpu_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

echo "======================================================================"
echo "Job started on $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

# Load necessary modules
module purge
module load python/3.9.10/gcc-11.2.0
module load cuda/11.8.0/gcc-11.2.0

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
echo "Key packages:"
python -c "import torch; import numpy; print(f'PyTorch: {torch.__version__}'); print(f'NumPy: {numpy.__version__}')"
echo ""

# Change to the directory where your script is located
cd $WORKDIR/bdrp

# Run the prediction script
echo "Starting training on GPU..."
python predict_links.py

echo ""
echo "Job completed at $(date)"

