#!/bin/bash
set -e

echo "========================================="
echo "Downloading model checkpoint from Azure"
echo "========================================="

AZURE_SAS_TOKEN="${1:-}"
if [ -z "$AZURE_SAS_TOKEN" ]; then
    echo "ERROR: AZURE_SAS_TOKEN not provided"
    exit 1
fi

MODEL_URL="https://vqastorage6305.blob.core.windows.net/models/checkpoint_17_Ahmed_768_new.pth.tar"
MODEL_FILE="app/checkpoint_17_Ahmed_768_new.pth.tar"

echo "Downloading from: $MODEL_URL"
curl -L --max-time 600 --retry 5 --retry-delay 10 \
    -o "$MODEL_FILE" \
    "${MODEL_URL}?${AZURE_SAS_TOKEN}"

MODEL_SIZE=$(stat -c%s "$MODEL_FILE" 2>/dev/null || echo 0)
echo "Downloaded file size: $MODEL_SIZE bytes ($((MODEL_SIZE / 1048576)) MB)"

if [ "$MODEL_SIZE" -lt 500000000 ]; then
    echo "ERROR: File is too small (expected >= 500MB)"
    echo "First 500 bytes:"
    head -c 500 "$MODEL_FILE" || true
    exit 1
fi

echo "✓ Model downloaded successfully!"
echo "========================================="
