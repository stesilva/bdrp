#!/bin/bash
# Complete setup script for running predict_links.py on Ruche
# This script will upload your project files to Ruche

USER="silvas"
HOST="ruche.mesocentre.universite-paris-saclay.fr"
REMOTE_DIR="/workdir/silvas/bdrp"

echo "=========================================="
echo "Setting up BDRP project on Ruche cluster"
echo "=========================================="
echo ""
echo "This script will upload your project to: $USER@$HOST:$REMOTE_DIR"
echo "You will be prompted for your password: bdrpNSS2025"
echo ""

# Step 1: Create remote directory
echo "Step 1: Creating remote directory..."
ssh "$USER@$HOST" "mkdir -p $REMOTE_DIR"

# Step 2: Upload project files
echo ""
echo "Step 2: Uploading project files..."
echo "This may take a few minutes..."
echo "Excluding: venv, __pycache__, *.log, *.pt files"

# Upload essential files (excluding large/unnecessary files)
rsync -P -r --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.log' \
    --exclude='*.pt' \
    --exclude='.git' \
    --exclude='nl27k_live_output.log' \
    ./ "$USER@$HOST:$REMOTE_DIR/"

# Step 3: Upload data directories separately (they might be large)
if [ -d "nl27k" ]; then
    echo ""
    echo "Step 3: Uploading nl27k dataset..."
    echo "This may take a while if the dataset is large..."
    rsync -P -r nl27k/ "$USER@$HOST:$REMOTE_DIR/nl27k/"
fi

if [ -d "cn15k" ]; then
    echo ""
    echo "Uploading cn15k dataset..."
    rsync -P -r cn15k/ "$USER@$HOST:$REMOTE_DIR/cn15k/"
fi

echo ""
echo "=========================================="
echo "Upload complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Connect to Ruche: ./connect_remote.sh"
echo "2. Navigate to project: cd /workdir/silvas/bdrp"
echo "3. Set up Python environment:"
echo "   module load python/3.x  # Check available versions"
echo "   python -m venv venv"
echo "   source venv/bin/activate"
echo "   pip install torch numpy tqdm"
echo "4. Submit job: sbatch run_predict_links_ruche.sh"
echo ""
echo "To check job status: squeue -u $USER"
echo "To view output: tail -f rgcn_nl27k_*.out"

