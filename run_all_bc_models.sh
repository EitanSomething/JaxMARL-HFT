#!/bin/bash

# Run all BC architecture combinations with stratified train/test split
# Architectures: rnn_base, rnn_wide, rnn_deep, transformer
# Policies: vwap, twap
# Modes: cold_start (BC only), hot_start (BC + RL warm)

cd /home/eitant/Documents/School/ML-Capstone/JaxMARL-HFT
export PYTHONPATH="$(pwd):$PYTHONPATH"

PYTHON=/home/eitant/Documents/School/ML-Capstone/.venv/bin/python

# Parse command line arguments
MODE="cold_start"
if [ $# -gt 0 ]; then
    MODE=$1
fi

if [ "$MODE" != "cold_start" ] && [ "$MODE" != "hot_start" ]; then
    echo "Usage: $0 [cold_start|hot_start]"
    echo "  cold_start  - Run BC training only (default)"
    echo "  hot_start   - Run BC followed by RL warm start training"
    exit 1
fi

echo "Training mode: $MODE"
echo ""

# Arrays of architectures and policies
ARCHITECTURES=("rnn" "rnn_wide" "rnn_deep" "transformer")
POLICIES=("vwap")

# Counter for tracking
TOTAL=0
CURRENT=0

# Calculate total runs
for arch in "${ARCHITECTURES[@]}"; do
  for policy in "${POLICIES[@]}"; do
    TOTAL=$((TOTAL + 1))
  done
done

echo "Running $TOTAL BC training combinations with stratified splits..."
echo "=================================================="

# Run all combinations
for arch in "${ARCHITECTURES[@]}"; do
  for policy in "${POLICIES[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "[$CURRENT/$TOTAL] Starting: Architecture=$arch, Policy=$policy, Mode=$MODE"
    echo "=================================================="
    
    # Build architecture-specific parameters
    ARCH_PARAMS=""
    if [ "$arch" = "rnn_wide" ]; then
      ARCH_PARAMS="++WIDE_FACTOR=2"
    elif [ "$arch" = "rnn_deep" ]; then
      ARCH_PARAMS="++DEEP_LAYERS=3"
    elif [ "$arch" = "transformer" ]; then
      ARCH_PARAMS="++TRANSFORMER_MODEL_DIM=128 ++TRANSFORMER_NUM_LAYERS=2 ++TRANSFORMER_NUM_HEADS=4 ++TRANSFORMER_MLP_DIM=256"
    fi
    
    # Create run name
    RUN_NAME="bc-${arch}-${policy}"
    
    # Construct checkpoint directory path for BC
    BC_CKPT_DIR="/home/eitant/Documents/School/ML-Capstone/JaxMARL-HFT/checkpoints/MARLCheckpoints/BC-Comparison-Apple/${RUN_NAME}-AAPL-MultiYear"
    
    # ============ COLD START: BC Training ============
    echo "Running BC training (cold start)..."
    XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
    $PYTHON gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py \
      --config-name=ippo_rnn_JAXMARL_exec \
      TRAINING_MODE=bc \
      +EXPERT_POLICY=$policy \
      ARCHITECTURE=$arch \
      $ARCH_PARAMS \
      WANDB_MODE=online \
      ENTITY=eitansomething-n-a \
      PROJECT=BC-Comparison-Apple-Test9 \
      +wandb.name=$RUN_NAME-AAPL-MultiYear \
      TimePeriod=\'2012,2019,2020,2021\' \
      EvalTimePeriod=\'2012,2019,2020,2021\' \
      SWEEP_PARAMETERS=null \
      ++BC_EPISODES=128 \
      ++BC_EPOCHS=40 \
      ++BC_BATCH_SIZE=64 \
      ++BC_LR=0.0001 \
      ++BC_MIN_EPOCHS=10 \
      ++BC_EARLY_STOP_PATIENCE=8 \
      ++NUM_ENVS=16 \
      ++NUM_STEPS=128 \
      ++LR=[0.0003] \
      ++ANNEAL_LR=[true] \
      ++GRU_HIDDEN_DIM=128 \
      ++FC_DIM_SIZE=128 \
      +world_config.alphatradePath=/home/eitant/Documents/School/ML-Capstone/JaxMARL-HFT \
      +world_config.dataPath=/home/eitant/Documents/School/ML-Capstone/data \
      +world_config.stock=AAPL \
      +world_config.timePeriod=\'2012,2019,2020,2021\' \
      +world_config.book_depth=5 \
      '+dict_of_agents_configs.Execution.observation_space=engineered_with_obi'
    
    BC_STATUS=$?
    if [ $BC_STATUS -ne 0 ]; then
      echo "⚠ BC training [$CURRENT/$TOTAL] failed with status $BC_STATUS"
      continue
    else
      echo "✓ BC training [$CURRENT/$TOTAL] completed successfully"
    fi
    
    # ============ HOT START: RL Warm Training (if selected) ============
    if [ "$MODE" = "hot_start" ]; then
      echo ""
      echo "Running RL warm start training (hot start)..."
      RL_RUN_NAME="${RUN_NAME}-rl-warm"
      
      XLA_PYTHON_CLIENT_ALLOCATOR=platform \
      XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
      $PYTHON gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py \
        --config-name=ippo_rnn_JAXMARL_exec \
        TRAINING_MODE=rl_warm \
        +WARMSTART_CHECKPOINT_DIR=$BC_CKPT_DIR \
        +EXPERT_POLICY=$policy \
        ARCHITECTURE=$arch \
        $ARCH_PARAMS \
        WANDB_MODE=online \
        ENTITY=eitansomething-n-a \
        PROJECT=RL-Warm-Apple \
        +wandb.name=$RL_RUN_NAME-AAPL-MultiYear \
        TimePeriod=\'2012,2019,2020,2021\' \
        EvalTimePeriod=\'2012,2019,2020,2021\' \
        SWEEP_PARAMETERS=null \
        ++NUM_ENVS=16 \
        ++NUM_STEPS=128 \
        ++LR=[0.0001] \
        ++ANNEAL_LR=[true] \
        ++GRU_HIDDEN_DIM=128 \
        ++FC_DIM_SIZE=128 \
        +world_config.alphatradePath=/home/eitant/Documents/School/ML-Capstone/JaxMARL-HFT \
        +world_config.dataPath=/home/eitant/Documents/School/ML-Capstone/data \
        +world_config.stock=AAPL \
        +world_config.timePeriod=\'2012,2019,2020,2021\' \
        +world_config.book_depth=5 \
        '+dict_of_agents_configs.Execution.observation_space=engineered_with_obi'
      
      RL_STATUS=$?
      if [ $RL_STATUS -ne 0 ]; then
        echo "⚠ RL warm start [$CURRENT/$TOTAL] failed with status $RL_STATUS"
      else
        echo "✓ RL warm start [$CURRENT/$TOTAL] completed successfully"
      fi
    fi
  done
done

echo ""
echo "=================================================="
echo "All $TOTAL BC training runs completed!"
if [ "$MODE" = "hot_start" ]; then
  echo "Mode: Hot Start (BC + RL Warm)"
  echo "Metrics saved to W&B projects: BC-Comparison-Apple, RL-Warm-Apple"
else
  echo "Mode: Cold Start (BC only)"
  echo "Metrics saved to W&B project: BC-Comparison-Apple"
fi
