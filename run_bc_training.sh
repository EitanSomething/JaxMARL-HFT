#!/bin/bash

# BC Training on AMZN 2012 data with VWAP expert policy
# Properly formatted with per-step slippage tracking

cd /home/eitant/Documents/School/ML-Capstone/JaxMARL-HFT

# Set PYTHONPATH to include current directory
export PYTHONPATH="$(pwd):$PYTHONPATH"

python3 gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py \
  --config-name=ippo_rnn_JAXMARL_exec \
  TRAINING_MODE=bc \
  +EXPERT_POLICY=vwap \
  WANDB_MODE=online \
  ENTITY=eitansomething-n-a \
  TimePeriod=2012 \
  EvalTimePeriod=2012 \
  SWEEP_PARAMETERS=null \
  ++BC_EPISODES=32 \
  ++BC_EPOCHS=20 \
  ++BC_BATCH_SIZE=256 \
  ++NUM_ENVS=64 \
  ++NUM_STEPS=32 \
  ++GRU_HIDDEN_DIM=128 \
  ++FC_DIM_SIZE=128 \
  +world_config.alphatradePath=/home/eitant/Documents/School/ML-Capstone/JaxMARL-HFT \
  +world_config.dataPath=/home/eitant/Documents/School/ML-Capstone/data \
  +world_config.stock=AMZN \
  +world_config.timePeriod=2012
