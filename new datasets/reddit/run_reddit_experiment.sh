#!/bin/bash
# Run Bayesian RGCN on Reddit Hyperlink Network dataset

# Download Reddit data if not already present
REDDIT_DATA_DIR="../../data/reddit"
REDDIT_TSV="../../data/reddit_raw/soc-redditHyperlinks-body.tsv"

# Step 1: Convert Reddit data to expected format
if [ ! -f "$REDDIT_DATA_DIR/train.tsv" ]; then
    echo "Converting Reddit data..."
    if [ ! -f "$REDDIT_TSV" ]; then
        echo "ERROR: Reddit TSV file not found at $REDDIT_TSV"
        echo "Please download it from: https://snap.stanford.edu/data/soc-RedditHyperlinks.html"
        echo "And place it in ../../data/reddit_raw/"
        exit 1
    fi
    
    python convert_reddit_data.py \
        --input "$REDDIT_TSV" \
        --output "$REDDIT_DATA_DIR" \
        --use-sentiment-relation \
        --use-properties-confidence \
        --train-ratio 0.8 \
        --val-ratio 0.1
fi

# Step 2: Run Bayesian RGCN
echo "Training Bayesian RGCN on Reddit dataset..."
python main_bayesian.py \
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

echo "Experiment complete!"
