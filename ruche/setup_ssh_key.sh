#!/bin/bash
# Setup SSH key authentication for Ruche (one-time setup)
# This will copy your public key to the server so you don't need to enter password each time

USER="silvas"
HOST="ruche.mesocentre.universite-paris-saclay.fr"
KEY_FILE="$HOME/.ssh/id_ed25519_ruche.pub"

if [ ! -f "$KEY_FILE" ]; then
    echo "Error: SSH key not found at $KEY_FILE"
    exit 1
fi

echo "Setting up SSH key authentication for $USER@$HOST"
echo "You will be prompted for your password (bdrpNSS2025) this one time"
echo ""

# Copy public key to server
ssh-copy-id -i "$KEY_FILE" "$USER@$HOST"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ SSH key setup complete!"
    echo "You can now connect without entering a password:"
    echo "  ssh $USER@$HOST"
else
    echo ""
    echo "Setup failed. You can manually copy the key:"
    echo "  cat $KEY_FILE | ssh $USER@$HOST 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys'"
fi

