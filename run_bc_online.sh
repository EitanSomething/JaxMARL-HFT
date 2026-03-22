#!/bin/bash
source /home/eitant/Documents/School/ML-Capstone/.venv/bin/activate
cd /home/eitant/Documents/School/ML-Capstone/JaxMARL-HFT
PYTHONPATH=. python3 gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py \
  --config-name ippo_rnn_JAXMARL_exec \
  TRAINING_MODE=bc \
  WANDB_MODE=online \
  ENTITY=eitansomething-n-a \
  TimePeriod=2012 \
  EvalTimePeriod=2012 \
  SWEEP_PARAMETERS=null \
  +EXPERT_POLICY=twap \
  +world_config.alphatradePath=/home/eitant/Documents/School/ML-Capstone/JaxMARL-HFT \
  +world_config.dataPath=/home/eitant/Documents/School/ML-Capstone/data \
  +world_config.stock=AMZN \
  +world_config.timePeriod=2012 \
  +world_config.book_depth=1
