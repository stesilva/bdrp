#!/bin/bash
# Download files/directories from Ruche cluster
# Usage: ./download_from_ruche.sh <remote_path> [local_path]
# Example: ./download_from_ruche.sh /workdir/silvas/results ./

USER="silvas"
HOST="ruche.mesocentre.universite-paris-saclay.fr"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <remote_path> [local_path]"
    echo "Example: $0 /workdir/silvas/results ./"
    exit 1
fi

REMOTE_PATH="$1"
LOCAL_PATH="${2:-./}"

echo "Downloading $USER@$HOST:$REMOTE_PATH to $LOCAL_PATH"
echo "You will be prompted for your password"

# Use rsync for better progress and resume capability
rsync -P -r -v "$USER@$HOST:$REMOTE_PATH" "$LOCAL_PATH"

echo ""
echo "Download complete!"

