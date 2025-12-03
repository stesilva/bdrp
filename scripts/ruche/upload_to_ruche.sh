#!/bin/bash
# Upload files/directories to Ruche cluster
# Usage: ./upload_to_ruche.sh <local_path> [remote_path]
# Example: ./upload_to_ruche.sh ./my_project /workdir/silvas/

USER="silvas"
HOST="ruche.mesocentre.universite-paris-saclay.fr"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <local_path> [remote_path]"
    echo "Example: $0 ./my_project /workdir/silvas/"
    exit 1
fi

LOCAL_PATH="$1"
REMOTE_PATH="${2:-/workdir/silvas/}"

echo "Uploading $LOCAL_PATH to $USER@$HOST:$REMOTE_PATH"
echo "You will be prompted for your password"

# Use rsync for better progress and resume capability
rsync -P -r -v "$LOCAL_PATH" "$USER@$HOST:$REMOTE_PATH"

echo ""
echo "Upload complete!"

