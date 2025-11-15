#!/bin/bash
# Script to connect to mesocentre remote server (Ruche cluster)
# User: silvas
# Password: bdrpNSS2025
# Hostname: ruche.mesocentre.universite-paris-saclay.fr

USER="silvas"
HOST="ruche.mesocentre.universite-paris-saclay.fr"

echo "Connecting to Ruche cluster..."
echo "Host: $HOST"
echo "User: $USER"
echo ""
echo "You will be prompted for your password: bdrpNSS2025"
echo ""

# Connect via SSH
ssh "$USER@$HOST"

