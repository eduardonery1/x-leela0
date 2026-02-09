#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt
pip install chess

echo ""
echo "=== Verifying data files ==="
for f in lc0/weights lc0/lc0_pytorch.py lc0/proto/net_pb2.py data/cache.json; do
    if [ -f "$f" ]; then
        echo "  OK: $f"
    else
        echo "  MISSING: $f"
        exit 1
    fi
done

echo ""
echo "=== Pre-downloading HuggingFace models ==="
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM

print('Downloading ChessGPT tokenizer...')
AutoTokenizer.from_pretrained('Waterhorse/chessgpt-base-v1')

print('Downloading gpt2 (default LLM)...')
AutoTokenizer.from_pretrained('gpt2')
AutoModelForCausalLM.from_pretrained('gpt2')

print('Done.')
"

echo ""
echo "=== Setup complete ==="
echo "Stage 1: python pretrain_contrastive.py --batch-size 32 --max-steps 20000"
echo "Stage 2: python train.py --batch-size 4 --pretrain checkpoints/pretrain_final.pt --max-steps 50000"
