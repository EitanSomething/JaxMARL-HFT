#!/bin/bash

# Quick test: Run 2 architectures with VWAP to verify setup
# Use smaller configs for faster iteration

set -e

PROJECT_DIR="/home/eitant/Documents/School/ML-Capstone/JaxMARL-HFT"
VENV_PATH="/home/eitant/Documents/School/ML-Capstone/.venv"

cd "$PROJECT_DIR"
export PYTHONPATH="$(pwd):$PYTHONPATH"
source "$VENV_PATH/bin/activate"

echo "========================================"
echo "Quick Test Run - VWAP Expert"
echo "========================================"
echo ""

# Test configurations (smaller for faster testing)
TEST_ARCHS=("rnn_base" "rnn_wide")

for ARCH in "${TEST_ARCHS[@]}"; do
  echo ">>> Testing architecture: $ARCH"
  
  python3 gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py \
    --config-name=ippo_rnn_JAXMARL_exec \
    TRAINING_MODE=bc \
    +EXPERT_POLICY=vwap \
    WANDB_MODE=online \
    ENTITY=eitansomething-n-a \
    TimePeriod=2012,2019,2020,2021 \
    EvalTimePeriod=2012 \
    SWEEP_PARAMETERS=null \
    ++BC_EPISODES=8 \
    ++BC_EPOCHS=5 \
    ++BC_BATCH_SIZE=256 \
    ++NUM_ENVS=32 \
    ++NUM_STEPS=16 \
    ++ARCHITECTURE=$ARCH \
    hydra.run.dir="results/test_runs/${ARCH}" \
    hydra.job.name="test_${ARCH}" \
    +world_config.alphatradePath=$PROJECT_DIR \
    +world_config.dataPath=/home/eitant/Documents/School/ML-Capstone/data \
    +world_config.stock=AAPL \
    +world_config.timePeriod='2012,2019,2020,2021' \
    +world_config.book_depth=5
  
  echo "✓ $ARCH completed"
  echo ""
done

echo "========================================"
echo "✓ Quick test complete!"
echo "Check W&B for results"
echo "========================================"
