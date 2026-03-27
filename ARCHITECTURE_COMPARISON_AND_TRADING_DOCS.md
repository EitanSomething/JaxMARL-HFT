# JaxMARL-HFT Architectures & Trading Options

## Neural Network Architectures

The framework provides four main agent neural network architectures, each designed to process Limit Order Book (LOB) data differently as they construct their policies. All models are initialized within `gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py`.

### 1. `rnn_base` (or `rnn`)
**Description:** The standard Recurrent Neural Network baseline. 
**Structure:**
- Processes environment observations through a single fully connected (Dense) layer matching the `FC_DIM_SIZE`.
- Passes the dense embeddings into a Gated Recurrent Unit (GRU) core represented by a `ScannedRNN` of size `GRU_HIDDEN_DIM`.
- Projects the hidden memory states out to final discrete policies via linear actor/critic networks.
**Use Case:** The most computationally and memory-efficient starting point; sets a baseline for learning temporal LOB dynamics.

### 2. `rnn_wide`
**Description:** A horizontally scaled version of the base GRU.
**Structure:**
- Increases both the initial feed-forward dimension and the GRU hidden memory state dimension by a specified `WIDE_FACTOR` (e.g., $2 \times$).
- Thus, if `FC_DIM_SIZE` is 128, the embedding becomes 256. 
**Use Case:** Allows the network to map wider arrays of independent signals per timestep without adding additional computational depth latency, providing a good balance between raw capacity and memory speed.

### 3. `rnn_deep`
**Description:** A vertically scaled deep feature extractor.
**Structure:**
- Before touching recurrent memory logic, observations are passed through several stacked Dense layers (ruled by `DEEP_LAYERS`).
- Uses $3$ deep layers by default.
**Use Case:** Best when the provided limits order book representations (`engineered` vs `basic` observation spaces) hold complex, highly nonlinear correlations that must be resolved into simpler abstractions *before* analyzing the time-series profile.

### 4. `transformer`
**Description:** An attention-based temporal model.
**Structure:**
- Replaces the `ScannedRNN` with self-attention Transformer blocks.
- Configurable via `TRANSFORMER_MODEL_DIM`, `TRANSFORMER_NUM_LAYERS`, `TRANSFORMER_NUM_HEADS`, and `TRANSFORMER_MLP_DIM`.
- Leverages parallelizable sequence look-backs rather than compressing past sequences into a single hidden GRU state.
**Use Case:** Useful for longer memory horizons or when sharp, distinct events deep in the sequence history (like an isolated large block print) are predictive heavily independent of recent smoothed volatility. Requires significantly more VRAM bandwidth than the RNN choices.

***

## Trading Operations & Options

`JaxMARL-HFT` divides tasks across two distinct agent types mapped by discrete configuration options.

### 1. Execution Agents (Buyer / Seller)
**Goal:** Liquidate or acquire a given target inventory (`task_size`) within a specific execution window (`episode_time`) while minimizing slippage against a benchmark.
- **Doom Penalty & Reference Prices:** If an agent fails to finish their full sequence of orders by the end of the timeline, the environment forces a fictional liquidation "doom trade" for the remaining quantity (`quant_left > 0`). The harshness of this penalty is defined by the `reference_price` setting, which determines if remaining bags evaluate terribly against the aggressive `far_touch` price or slightly more forgivingly against the `mid` price.
- **Action Spaces:** 
  - `fixed_quants_complex`: Discretized action spaces forcing the agent to select fixed size chunks across fixed price ticks in the LOB.
  - `twap` & `vwap`: Binary action spaces where the agent only decides whether to `HOLD` (`action=0`) or `BUY`/`EXECUTE` (`action=1`) aggressively at the far touch price. The scheduled volume to execute is determined strictly by the underlying algorithmic planner (mirroring the DNN setups).
- **Data Imbalance Mitigation:** In behavioral cloning contexts (like mapping to a TWAP/VWAP expert policy), the agent heavily favors holding over executing (roughly $\approx 90\%$ `HOLD` vs $10\%$ `EXECUTE`). To prevent the model from collapsing into a naive "always hold" policy, the PPO/Behavioral Cloning loss function dynamically calculates inverse frequency class weights on every batch:
  - $W_0 = \frac{N_1}{N_0 + N_1}$
  - $W_1 = \frac{N_0}{N_0 + N_1}$
  These weights are applied inversely to the log probabilities during loss computation, forcing the network gradient to pay significantly more attention to correctly catching the critical rare `EXECUTE` events.  
- **Pre-coded Expert & RL Planner Algorithms:** Used for policy shaping and Behavioral Cloning (BC).
  - **TWAP (Time-Weighted Average Price):** The internal schedule dynamically divides the remaining quantity equally over the remaining time steps. The policy simply triggers `HOLD` or `EXECUTE`.
  - **VWAP (Volume-Weighted Average Price):** The internal schedule tracks a predefined U-shaped intraday market volume profile. Triggering an `EXECUTE` action forces the agent to place a market order sized precisely to "catch up" to its expected cumulative fractional target.

### 2. Market Making Agents (Liquidity Providers)
**Goal:** Generate consistent profit by capturing the spread while simultaneously remaining as delta-neutral as possible to avoid unwanted directional inventory exposure.
- **Reward Functions:** Features options like `spooner_asym_damped2` which uses highly asymmetric inventory penalties (`inv_penalty_quadratic_factor`) to exponentially drag the agent's reward down if they start drifting heavily long or heavily short.
- **Unwinding Parameters:** Forces agents to recognize penalties tied to liquidating their inventory out at bad prices if they end trading windows carrying a net stack (`unwind_price_penalty`).
- **Observation Space Settings:** Often rely on the `basic` un-engineered features, learning their own custom limits inside the network to identify immediate term mispricing. 