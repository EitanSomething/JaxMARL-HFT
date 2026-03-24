# Architecture Comparison Scripts

Scripts for running BC training experiments across different model architectures and comparing results.

## Scripts

### 1. `run_quick_test.sh` - Quick Test Run (5 min)
Tests 2 architectures (rnn_base, rnn_wide) with VWAP expert for fast validation.

```bash
chmod +x run_quick_test.sh
./run_quick_test.sh
```

**Output:**
- Quick validation that training pipeline works
- 2 W&B runs to check
- ~5 minutes total

### 2. `run_architecture_sweep.sh` - Full Sweep (2-4 hours)
Runs all combinations:
- Architectures: `rnn_base`, `rnn_wide`, `rnn_deep`, `transformer` (4)
- Expert Policies: `vwap`, `twap` (2)
- **Total: 8 runs**

```bash
chmod +x run_architecture_sweep.sh
./run_architecture_sweep.sh
```

**Output:**
- `results/architecture_sweep/` - All run outputs
- `results/architecture_sweep/comparison_results.csv` - Metrics table
- Console summary with results grouped by expert policy

### 3. `compare_runs.py` - W&B Comparison Tool
Fetch runs from W&B and generate formatted comparison table.

```bash
python3 compare_runs.py \
  --entity eitansomething-n-a \
  --project exec_env_bc \
  --mode bc \
  --output architecture_comparison.csv
```

**Output:**
- Console table with formatted results
- `architecture_comparison.csv` - Full comparison
- Summary statistics (mean/std by architecture)

## Workflow

### Option A: Quick Validation
```bash
./run_quick_test.sh
# Wait 5 min, check results in W&B
```

### Option B: Full Comparison (Recommended)
```bash
./run_architecture_sweep.sh
# Watch runs progress in output
# Full results at results/architecture_sweep/comparison_results.csv
```

### Option C: Post-Run Analysis (After experiments complete)
```bash
python3 compare_runs.py
# Generates clean comparison table from W&B
```

## Configuration Tuning

Edit the shell scripts to adjust:

```bash
# In run_architecture_sweep.sh:
BC_EPISODES=16       # Increase for better estimates
BC_EPOCHS=10         # More epochs = longer training
BC_BATCH_SIZE=256    # Batch size
NUM_ENVS=32          # Parallel environments
NUM_STEPS=16         # Steps per rollout
```

### Preset Configs

**Fast (5 min per run):**
```
BC_EPISODES=8, BC_EPOCHS=5, NUM_ENVS=16, NUM_STEPS=8
```

**Standard (10 min per run):**
```
BC_EPISODES=16, BC_EPOCHS=10, NUM_ENVS=32, NUM_STEPS=16
```

**Thorough (20 min per run):**
```
BC_EPISODES=32, BC_EPOCHS=20, NUM_ENVS=64, NUM_STEPS=32
```

## Metrics Captured

| Metric | Description |
|--------|-------------|
| `accuracy` | % correct execute/hold predictions |
| `f1` | F1 score on execute action |
| `precision` | Precision of execute predictions |
| `recall` | Recall of execute predictions |
| `slippage_bps` | Execution slippage in basis points |

## Results Interpretation

### Expected Results

**VWAP vs TWAP:**
- VWAP: Better accuracy (following volume schedule reduces execution variability)
- TWAP: Simpler policy, potentially higher slippage

**Architecture Comparison:**
- `rnn_base`: Baseline (128 hidden, 128 FC)
- `rnn_wide`: Wider layers (256 hidden, 256 FC) - usually better
- `rnn_deep`: Stacked RNNs - risk of overfitting on small dataset
- `transformer`: Attention-based - longest training, potential for overfitting

### Example CSV Output

```
expert_policy,architecture,accuracy,f1,precision,slippage_bps
vwap,rnn_base,0.9453,0.8234,0.8567,-0.4521
vwap,rnn_wide,0.9612,0.8567,0.8901,-0.5234
twap,rnn_base,0.8234,0.7123,0.7234,0.6543
twap,rnn_wide,0.8567,0.7456,0.7612,0.5432
```

## Troubleshooting

### "Command not found"
```bash
chmod +x run_*.sh
```

### Runs not showing in W&B
- Check entity is set correctly: `ENTITY=eitansomething-n-a`
- Enable W&B: `WANDB_MODE=online`
- Verify W&B login: `wandb login`

### Out of memory
Reduce `NUM_ENVS` or `NUM_STEPS` in script

### Slow runs
- Decrease `BC_EPOCHS` for faster iteration
- Use `run_quick_test.sh` instead
- Can run architectures in parallel on multiple machines

## Advanced Usage

### Run specific architecture only
```bash
# Edit run_architecture_sweep.sh
ARCHITECTURES=("rnn_wide")  # Only test rnn_wide
EXPERT_POLICIES=("vwap")    # Only VWAP
./run_architecture_sweep.sh
```

### Extract metrics from local logs only
```bash
python3 compare_runs.py --mode="" 2>/dev/null || \
  grep -h "val_accuracy\|slippage" results/*/train.log
```

### Run on GPU cluster (parallel)
```bash
# Start run_architecture_sweep.sh on multiple nodes
# Each runs subset of architectures
for ARCH in rnn_base rnn_wide; do
  ARCHITECTURES=("$ARCH") ./run_architecture_sweep.sh &
done
wait
```
