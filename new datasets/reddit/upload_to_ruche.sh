#!/bin/bash
# Upload Reddit experiment to Ruche cluster
# Usage: ./upload_to_ruche.sh

USER="silvas"
HOST="ruche.mesocentre.universite-paris-saclay.fr"
REMOTE_BASE="/workdir/silvas/bdrp"
REMOTE_DIR="$REMOTE_BASE/newdatasets/reddit"

echo "=========================================="
echo "Uploading Reddit experiment to Ruche"
echo "=========================================="
echo ""
echo "This script will upload to: $USER@$HOST:$REMOTE_DIR"
echo "You will be prompted for your password"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Step 1: Create remote directory structure
echo "Step 1: Creating remote directory structure..."
ssh "$USER@$HOST" "mkdir -p $REMOTE_DIR"

# Step 2: Upload experiment files
echo ""
echo "Step 2: Uploading experiment files..."
echo "This may take a few minutes..."

# Upload essential files (excluding large/unnecessary files)
rsync -P -r --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.log' \
    --exclude='*.pt' \
    --exclude='.git' \
    --exclude='*.ipynb_checkpoints' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='analyze_bayesian_weights.ipynb' \
    ./ "$USER@$HOST:$REMOTE_DIR/"

# Step 3: Upload converted dataset if it exists
if [ -d "reddit_converted" ]; then
    echo ""
    echo "Step 3: Uploading converted Reddit dataset..."
    echo "This may take a while if the dataset is large..."
    rsync -P -r reddit_converted/ "$USER@$HOST:$REMOTE_DIR/reddit_converted/"
fi

# Step 4: Check if requirements.txt exists in parent directory and upload if needed
if [ -f "../../requirements.txt" ]; then
    echo ""
    echo "Step 4: Uploading requirements.txt..."
    ssh "$USER@$HOST" "mkdir -p $REMOTE_BASE"
    rsync -P ../../requirements.txt "$USER@$HOST:$REMOTE_BASE/requirements.txt"
fi

echo ""
echo "=========================================="
echo "Upload complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Connect to Ruche: ssh $USER@$HOST"
echo "2. Navigate to: cd '$REMOTE_DIR'"
echo "3. Set up environment (if needed):"
echo "   module load python/3.9.10/gcc-11.2.0"
echo "   module load cuda/11.8.0/gcc-11.2.0"
echo "   python -m venv venv"
echo "   source venv/bin/activate"
echo "   pip install torch torch-geometric numpy pandas tqdm scipy"
echo "   # Or install from requirements: pip install -r ../../requirements.txt"
echo "4. Submit job: sbatch run_reddit_experiment_ruche.sh"
echo ""
echo "To check job status: squeue -u $USER"
echo "To view output: tail -f reddit_bayesian_*.out"
echo "To cancel job: scancel <job_id>"
