#!/bin/bash

# Run BC training sweep across all architectures and expert policies
# Captures metrics and generates comparison

set -e

PROJECT_DIR="/home/eitant/Documents/School/ML-Capstone/JaxMARL-HFT"
RESULTS_DIR="$PROJECT_DIR/results/architecture_sweep"
VENV_PATH="/home/eitant/Documents/School/ML-Capstone/.venv"

cd "$PROJECT_DIR"
export PYTHONPATH="$(pwd):$PYTHONPATH"
source "$VENV_PATH/bin/activate"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Configuration
ARCHITECTURES=("rnn_base" "rnn_wide" "rnn_deep" "transformer")
EXPERT_POLICIES=("vwap" "twap")
BC_EPISODES=16
BC_EPOCHS=10
BC_BATCH_SIZE=64
NUM_ENVS=16
NUM_STEPS=16

TOTAL_RUNS=$((${#ARCHITECTURES[@]} * ${#EXPERT_POLICIES[@]}))
CURRENT_RUN=0

echo "========================================"
echo "Architecture Comparison Sweep"
echo "========================================"
echo "Total runs: $TOTAL_RUNS"
echo "Architectures: ${ARCHITECTURES[@]}"
echo "Expert Policies: ${EXPERT_POLICIES[@]}"
echo "Results dir: $RESULTS_DIR"
echo "========================================"
echo ""

# Run all combinations
for EXPERT in "${EXPERT_POLICIES[@]}"; do
  for ARCH in "${ARCHITECTURES[@]}"; do
    CURRENT_RUN=$((CURRENT_RUN + 1))
    RUN_NAME="${EXPERT}_${ARCH}"
    RUN_DIR="$RESULTS_DIR/${RUN_NAME}"
    
    echo "[$CURRENT_RUN/$TOTAL_RUNS] Running: $RUN_NAME"
    echo "Start time: $(date)"
    
    mkdir -p "$RUN_DIR"
    
    # Run training
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
    python3 gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py \
      --config-name=ippo_rnn_JAXMARL_exec \
      TRAINING_MODE=bc \
      +EXPERT_POLICY=$EXPERT \
      WANDB_MODE=online \
      ENTITY=eitansomething-n-a \
      TimePeriod=\'2012,2019,2020,2021\' \
      EvalTimePeriod=\'2012,2019,2020,2021\' \
      SWEEP_PARAMETERS=null \
      ++BC_EPISODES=$BC_EPISODES \
      ++BC_EPOCHS=$BC_EPOCHS \
      ++BC_BATCH_SIZE=$BC_BATCH_SIZE \
      ++NUM_ENVS=$NUM_ENVS \
      ++NUM_STEPS=$NUM_STEPS \
      ++ARCHITECTURE=$ARCH \
      hydra.run.dir="$RUN_DIR/hydra_outputs" \
      hydra.job.name=$RUN_NAME \
      +world_config.alphatradePath=$PROJECT_DIR \
      +world_config.dataPath=/home/eitant/Documents/School/ML-Capstone/data \
      +world_config.stock=AAPL \
      +world_config.timePeriod=\'2012,2019,2020,2021\' \
      +world_config.book_depth=5 \
      2>&1 | tee "$RUN_DIR/train.log"
    
    echo "✓ Completed: $RUN_NAME at $(date)"
    echo ""
    
  done
done

echo "========================================"
echo "Generating Comparison Report"
echo "========================================"
echo ""

# Extract metrics and create comparison table
python3 << 'PYTHON_EOF'
import os
import re
import json
import csv
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("/home/eitant/Documents/School/ML-Capstone/JaxMARL-HFT/results/architecture_sweep")
COMPARISON_FILE = RESULTS_DIR / "comparison_results.csv"

# Extract key metrics from run logs
def extract_metrics_from_log(log_file):
    metrics = {}
    
    if not log_file.exists():
        return metrics
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            
            # Parse wandb metric lines: "wandb:      bc/agent_EXE/val_accuracy 0.61844"
            for line in lines:
                if 'wandb:' not in line:
                    continue
                
                # Extract metric name and value
                # Format: "wandb:      <metric_name> <value>"
                match = re.search(r'wandb:\s+([\w/]+)\s+([\d.-]+)', line)
                if not match:
                    continue
                
                metric_name = match.group(1)
                value = float(match.group(2))
                
                # Map metric names to simple keys
                if 'val_accuracy' in metric_name:
                    metrics['accuracy'] = value
                elif 'val_f1' in metric_name:
                    metrics['f1'] = value
                elif 'val_precision' in metric_name:
                    metrics['precision'] = value
                elif 'val_recall' in metric_name:
                    metrics['recall'] = value
                elif 'slippage_bps' in metric_name or 'price_slippage' in metric_name:
                    metrics['slippage_bps'] = value
    
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
    
    return metrics

# Collect all run results
results = []
for run_dir in sorted(RESULTS_DIR.glob("*")):
    if not run_dir.is_dir():
        continue
    
    run_name = run_dir.name
    parts = run_name.split('_', 1)
    if len(parts) != 2:
        continue
    
    expert_policy, architecture = parts
    log_file = run_dir / "train.log"
    
    metrics = extract_metrics_from_log(log_file)
    
    result = {
        'expert_policy': expert_policy,
        'architecture': architecture,
        'run_name': run_name,
        'log_exists': log_file.exists(),
        'accuracy': metrics.get('accuracy'),
        'f1': metrics.get('f1'),
        'precision': metrics.get('precision'),
        'recall': metrics.get('recall'),
        'slippage_bps': metrics.get('slippage_bps'),
    }
    results.append(result)

# Write CSV
if results:
    fieldnames = ['expert_policy', 'architecture', 'run_name', 'log_exists', 
                  'accuracy', 'f1', 'precision', 'recall', 'slippage_bps']
    
    with open(COMPARISON_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✓ Comparison file saved: {COMPARISON_FILE}")
    print("")
    print("="*100)
    print("RESULTS SUMMARY")
    print("="*100)
    
    # Group by expert policy
    by_expert = defaultdict(list)
    for r in results:
        by_expert[r['expert_policy']].append(r)
    
    for expert in sorted(by_expert.keys()):
        print(f"\n{expert.upper()} Expert Policy:")
        print("-" * 100)
        print(f"{'Architecture':<15} {'Accuracy':<15} {'F1':<15} {'Precision':<15} {'Slippage (bps)':<15}")
        print("-" * 100)
        
        for r in sorted(by_expert[expert], key=lambda x: x['architecture']):
            acc = f"{r['accuracy']:.4f}" if r['accuracy'] is not None else "N/A"
            f1 = f"{r['f1']:.5f}" if r['f1'] is not None else "N/A"
            prec = f"{r['precision']:.5f}" if r['precision'] is not None else "N/A"
            slip = f"{r['slippage_bps']:.6f}" if r['slippage_bps'] is not None else "N/A"
            
            print(f"{r['architecture']:<15} {acc:<15} {f1:<15} {prec:<15} {slip:<15}")
    
    print("\n" + "="*100)
else:
    print("No results found. Check that runs completed successfully.")

PYTHON_EOF

echo ""
echo "✓ Sweep complete!"
echo "Review: $RESULTS_DIR/comparison_results.csv"
