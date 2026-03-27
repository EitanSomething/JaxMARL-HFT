#!/bin/bash

# Ensure we are in the right directory
cd /home/eitant/Documents/School/ML-Capstone/JaxMARL-HFT
export PYTHONPATH="$(pwd):$PYTHONPATH"

PYTHON=/home/eitant/Documents/School/ML-Capstone/.venv/bin/python
PROJECT_BC="BC-Apple-Final3"
PROJECT_RL="BC-Apple-Final3"

ARCHITECTURES=("rnn")
POLICIES=("twap")

# Use a standard string for periods to avoid Lexer issues
PERIODS="2012,2019,2020,2021"
for arch in "${ARCHITECTURES[@]}"; do
  for policy in "${POLICIES[@]}"; do
    
    ARCH_PARAMS=""
    case $arch in
      "rnn_wide")    ARCH_PARAMS="++WIDE_FACTOR=2" ;;
      "rnn_deep")    ARCH_PARAMS="++DEEP_LAYERS=3" ;;
      "transformer") ARCH_PARAMS="++TRANSFORMER_MODEL_DIM=128 ++TRANSFORMER_NUM_LAYERS=2 ++TRANSFORMER_NUM_HEADS=4 ++TRANSFORMER_MLP_DIM=256" ;;
    esac

    RUN_NAME="bc-${arch}-${policy}"
    RUN_FULL_NAME="${RUN_NAME}-AAPL-MultiYear"
    BC_CKPT_DIR="/home/eitant/Documents/School/ML-Capstone/JaxMARL-HFT/checkpoints/MARLCheckpoints/${PROJECT_BC}/${RUN_FULL_NAME}"

    echo "PHASE 1: BC Training ($arch / $policy)"
    
    # Note: Removed trailing spaces after backslashes and simplified quoting
    XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
    HYDRA_FULL_ERROR=1 \
    $PYTHON gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py \
      --config-name=ippo_rnn_JAXMARL_exec \
      TRAINING_MODE=bc \
      +EXPERT_POLICY=$policy \
      ARCHITECTURE=$arch \
      $ARCH_PARAMS \
      WANDB_MODE=online \
      ENTITY=eitansomething-n-a \
      PROJECT=$PROJECT_BC \
      +wandb.name="$RUN_FULL_NAME" \
      "TimePeriod=\"$PERIODS\"" \
      "EvalTimePeriod=\"$PERIODS\"" \
      SWEEP_PARAMETERS=null \
      ++BC_EPISODES=128 \
      ++BC_EPOCHS=40 \
      ++BC_BATCH_SIZE=64 \
      ++BC_LR=0.0001 \
      ++NUM_ENVS=16 \
      ++NUM_STEPS=128 \
      ++GRU_HIDDEN_DIM=128 \
      ++FC_DIM_SIZE=128 \
      +world_config.alphatradePath=/home/eitant/Documents/School/ML-Capstone/JaxMARL-HFT \
      +world_config.dataPath=/home/eitant/Documents/School/ML-Capstone/data \
      +world_config.stock=AAPL \
      +world_config.book_depth=5 \
      "+dict_of_agents_configs.Execution.observation_space=engineered_with_obi"

    if [ $? -ne 0 ]; then
      echo "BC Phase failed. Skipping RL phase."
      continue
    fi

    # Check if checkpoint was created
    if [ ! -d "$BC_CKPT_DIR" ] || [ -z "$(find "$BC_CKPT_DIR" -type f -name '*.eqx' 2>/dev/null | head -1)" ]; then
      echo "No checkpoint found in $BC_CKPT_DIR. Skipping RL phase."
      continue
    fi

    echo "PHASE 2: RL Warm Start ($arch / $policy)"
    
    XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
    HYDRA_FULL_ERROR=1 \
    $PYTHON gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py \
      --config-name=ippo_rnn_JAXMARL_exec \
      TRAINING_MODE=rl_warm \
      +WARMSTART_CHECKPOINT_DIR="$BC_CKPT_DIR" \
      +EXPERT_POLICY=$policy \
      ARCHITECTURE=$arch \
      $ARCH_PARAMS \
      WANDB_MODE=online \
      ENTITY=eitansomething-n-a \
      PROJECT=$PROJECT_RL \
      +wandb.name="${RUN_NAME}-rl-warm-AAPL-MultiYear" \
      "TimePeriod=\"$PERIODS\"" \
      "EvalTimePeriod=\"$PERIODS\"" \
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
      +world_config.book_depth=5 \
      "+dict_of_agents_configs.Execution.observation_space=engineered_with_obi"
  done
done