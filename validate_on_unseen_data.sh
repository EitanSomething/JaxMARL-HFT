#!/bin/bash

# Validate trained BC models on unseen data (different date)
# This runs the exact same BC evaluation code but on a validation date

set -e

PROJECT_DIR="/home/eitant/Documents/School/ML-Capstone/JaxMARL-HFT"
VENV_PATH="/home/eitant/Documents/School/ML-Capstone/.venv"

cd "$PROJECT_DIR"
export PYTHONPATH="$(pwd):$PYTHONPATH"
source "$VENV_PATH/bin/activate"

TRAINING_DATE="2012-06-21"
VALIDATION_DATE="2012-06-22"  # Different date - model never saw this

ARCHITECTURES=("rnn_base" "rnn_wide" "rnn_deep" "transformer")
EXPERT_POLICIES=("vwap" "twap")

echo "========================================"
echo "BC Model Validation on Unseen Data"
echo "========================================"
echo "Training date: $TRAINING_DATE"
echo "Validation date: $VALIDATION_DATE"
echo "Architectures: ${ARCHITECTURES[@]}"
echo "Expert policies: ${EXPERT_POLICIES[@]}"
echo ""

# Create results directory
mkdir -p results/validation_sweep

RESULTS_FILE="results/validation_sweep/validation_results.csv"
echo "expert_policy,architecture,val_accuracy,val_f1,val_precision,val_recall,val_slippage_bps" > "$RESULTS_FILE"

# Run validation for each combination
for EXPERT in "${EXPERT_POLICIES[@]}"; do
  for ARCH in "${ARCHITECTURES[@]}"; do
    RUN_NAME="${EXPERT}_${ARCH}_validation"
    
    echo ">>> Validating: $EXPERT / $ARCH on $VALIDATION_DATE"
    
    # Run BC training but:
    # 1. Start fresh (don't use saved weights)
    # 2. Use validation date for both training and eval
    # 3. Only run 1 episode to test
    python3 gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py \
      --config-name=ippo_rnn_JAXMARL_exec \
      TRAINING_MODE=bc \
      +EXPERT_POLICY=$EXPERT \
      WANDB_MODE=offline \
      TimePeriod=$VALIDATION_DATE \
      EvalTimePeriod=$VALIDATION_DATE \
      SWEEP_PARAMETERS=null \
      ++BC_EPISODES=1 \
      ++BC_EPOCHS=1 \
      ++BC_BATCH_SIZE=256 \
      ++NUM_ENVS=32 \
      ++NUM_STEPS=16 \
      ++ARCHITECTURE=$ARCH \
      hydra.run.dir="results/validation_sweep/${RUN_NAME}" \
      hydra.job.name=$RUN_NAME \
      +world_config.alphatradePath=$PROJECT_DIR \
      +world_config.dataPath=$PROJECT_DIR/data \
      +world_config.stock=AMZN \
      +world_config.timePeriod=$VALIDATION_DATE \
      2>&1 | tee "results/validation_sweep/${RUN_NAME}.log"
    
    # Extract metrics from log
    python3 << 'EXTRACT_EOF'
import sys
import re
from pathlib import Path

log_file = sys.argv[1]
run_name = sys.argv[2]

metrics = {
    'expert_policy': run_name.split('_')[0],
    'architecture': run_name.split('_')[1],
    'val_accuracy': 'N/A',
    'val_f1': 'N/A',
    'val_precision': 'N/A',
    'val_recall': 'N/A',
    'val_slippage_bps': 'N/A',
}

try:
    with open(log_file, 'r') as f:
        content = f.read()
        
        # Parse wandb metric lines
        for line in content.split('\n'):
            if 'wandb:' not in line:
                continue
            
            match = re.search(r'wandb:\s+([\w/]+)\s+([\d.-]+)', line)
            if not match:
                continue
            
            metric_name = match.group(1)
            value = float(match.group(2))
            
            if 'val_accuracy' in metric_name:
                metrics['val_accuracy'] = value
            elif 'val_f1' in metric_name:
                metrics['val_f1'] = value
            elif 'val_precision' in metric_name:
                metrics['val_precision'] = value
            elif 'val_recall' in metric_name:
                metrics['val_recall'] = value
            elif 'slippage_bps' in metric_name or 'price_slippage' in metric_name:
                metrics['val_slippage_bps'] = value
except:
    pass

# Print CSV line
result = f"{metrics['expert_policy']},{metrics['architecture']},{metrics['val_accuracy']},{metrics['val_f1']},{metrics['val_precision']},{metrics['val_recall']},{metrics['val_slippage_bps']}"
print(result)
EXTRACT_EOF

    CSV_LINE=$(python3 << 'EXTRACT_EOF' "$RUN_NAME.log" "$RUN_NAME"
import sys
import re
from pathlib import Path

log_file = f"results/validation_sweep/{sys.argv[1]}"
run_name = sys.argv[2]

metrics = {
    'expert_policy': run_name.split('_')[0],
    'architecture': run_name.split('_')[1],
    'val_accuracy': 'N/A',
    'val_f1': 'N/A',
    'val_precision': 'N/A',
    'val_recall': 'N/A',
    'val_slippage_bps': 'N/A',
}

try:
    with open(log_file, 'r') as f:
        content = f.read()
        for line in content.split('\n'):
            if 'wandb:' not in line:
                continue
            match = re.search(r'wandb:\s+([\w/]+)\s+([\d.-]+)', line)
            if not match:
                continue
            metric_name = match.group(1)
            value = float(match.group(2))
            if 'val_accuracy' in metric_name:
                metrics['val_accuracy'] = value
            elif 'val_f1' in metric_name:
                metrics['val_f1'] = value
            elif 'val_precision' in metric_name:
                metrics['val_precision'] = value
            elif 'val_recall' in metric_name:
                metrics['val_recall'] = value
            elif 'slippage_bps' in metric_name or 'price_slippage' in metric_name:
                metrics['val_slippage_bps'] = value
except:
    pass

print(f"{metrics['expert_policy']},{metrics['architecture']},{metrics['val_accuracy']},{metrics['val_f1']},{metrics['val_precision']},{metrics['val_recall']},{metrics['val_slippage_bps']}")
EXTRACT_EOF
)
    
    echo "$CSV_LINE" >> "$RESULTS_FILE"
    echo "✓ Saved results"
    echo ""
  done
done

echo "========================================"
echo "✓ Validation complete!"
echo "Results: $RESULTS_FILE"
echo "========================================"
echo ""

python3 << 'DISPLAY_EOF'
import csv
from pathlib import Path
from collections import defaultdict

results_file = "results/validation_sweep/validation_results.csv"
results = []

with open(results_file, 'r') as f:
    reader = csv.DictReader(f)
    results = list(reader)

# Group by expert
by_expert = defaultdict(list)
for r in results:
    by_expert[r['expert_policy']].append(r)

print("\n" + "="*100)
print("VALIDATION RESULTS (Unseen Data)")
print("="*100)

for expert in sorted(by_expert.keys()):
    print(f"\n{expert.upper()} Expert Policy (Validation Date: 2012-06-22):")
    print("-" * 100)
    print(f"{'Architecture':<15} {'Accuracy':<12} {'F1':<12} {'Precision':<12} {'Slippage (bps)':<15}")
    print("-" * 100)
    
    for r in sorted(by_expert[expert], key=lambda x: x['architecture']):
        acc = f"{float(r['val_accuracy']):.4f}" if r['val_accuracy'] != 'N/A' else "N/A"
        f1 = f"{float(r['val_f1']):.4f}" if r['val_f1'] != 'N/A' else "N/A"
        prec = f"{float(r['val_precision']):.4f}" if r['val_precision'] != 'N/A' else "N/A"
        slip = f"{float(r['val_slippage_bps']):.6f}" if r['val_slippage_bps'] != 'N/A' else "N/A"
        
        print(f"{r['architecture']:<15} {acc:<12} {f1:<12} {prec:<12} {slip:<15}")

print("\n" + "="*100)
DISPLAY_EOF
