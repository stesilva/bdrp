#!/bin/bash
#SBATCH --job-name=rgcn_link_pred
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu_short
#SBATCH --export=NONE
#SBATCH --propagate=NONE

# If you want to use GPU (recommended for larger graphs):
# Uncomment the following lines and comment out the cpu_short partition above
# #SBATCH --partition=gpu_short
# #SBATCH --gres=gpu:1
# #SBATCH --mem=32G

# Optional: Specify a project account (if you have multiple projects)
# #SBATCH --account=<YOUR_PROJECT>

# Optional: Email notifications
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=your.email@centralesupelec.fr

echo "======================================================================"
echo "Job started on $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "======================================================================"

# Load necessary modules
module purge
module load anaconda3/2020.02/gcc-9.2.0

# If using GPU, also load CUDA (check available versions with: module avail cuda)
# module load cuda/10.2.89/intel-19.0.3.199

# Activate anaconda environment
# Replace 'numpy-env' with your environment name (e.g., 'torch-env', 'pyg-env')
source activate torch-env

# Print environment info for debugging
echo ""
echo "Python version:"
python --version
echo ""
echo "Conda environment:"
conda info --envs | grep '*'
echo ""
echo "Key packages:"
pip list | grep -E "torch|geometric|numpy|scikit"
echo ""
echo "======================================================================"

# Navigate to the directory containing your script
# The job starts in the submission directory by default
cd $SLURM_SUBMIT_DIR

# Alternative: if your code is in a different location
# cd $HOME/rgcn_project
# cd $WORKDIR/rgcn_project

echo "Running R-GCN link prediction from: $(pwd)"
echo "Script: rgcn_link_prediction_test.py"
echo ""

# Run the python script
python rgcn_link_prediction_test.py

# Capture exit status
EXIT_STATUS=$?

# Print job statistics
echo ""
echo "======================================================================"
if [ $EXIT_STATUS -eq 0 ]; then
    echo "Job completed successfully on $(date)"
else
    echo "Job failed with exit code $EXIT_STATUS on $(date)"
fi
echo "======================================================================"
echo ""
echo "Job statistics (use 'seff $SLURM_JOB_ID' for detailed efficiency report):"
echo "CPU time used: $SECONDS seconds"

exit $EXIT_STATUS