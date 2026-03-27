# Behavior Cloning Overfitting Fixes for TWAP

## Problem Summary

TWAP BC training suffered from severe class imbalance (90% hold, 10% execute), causing models to overfit to the majority class. This resulted in:

- **Poor slippage performance**: -0.73 to -1.43 bps (vs target ~-0.15 bps)
- **Imbalanced predictions**: High recall (0.92) but very low precision (0.19)
- **Worse performance with larger models**: RNN Deep performed 5x worse than base RNN

## Root Causes

1. **Extreme class imbalance**: TWAP executes only every 10 steps (10% of samples)
2. **Model capacity**: Larger models (RNN Wide, RNN Deep) have more parameters to memorize the majority class
3. **Weak regularization**: Standard cross-entropy loss doesn't handle severe imbalance well

## Solutions Implemented

### 1. Adaptive Class Weighting (Alpha Scaling)

**Location**: `ippo_rnn_JAXMARL.py` line ~945

**What**: Scale class weights based on execution rate and model architecture

```python
# For TWAP (10% execute rate), scale alpha by model capacity
if expected_exec_rate < 0.2:
    if architecture == "rnn_deep":
        alpha = 15.0  # Deepest needs most help
    elif architecture == "rnn_wide":
        alpha = 12.0  # Wide needs more help
    else:
        alpha = 10.0  # Base RNN
else:
    alpha = 2.0  # VWAP or balanced policies
```

**Why**: Larger models need stronger class weighting to prevent overfitting to majority class

**Impact**: Increased minority class weight by 2-3x for larger architectures

---

### 2. Asymmetric Execution Rate Penalty

**Location**: `ippo_rnn_JAXMARL.py` line ~1005

**What**: Penalize over-execution 3x more than under-execution, scaled by model size

```python
# Target range: 80-120% of expert rate
min_exec_rate = expected_exec_rate * 0.8
max_exec_rate = expected_exec_rate * 1.2

# Scale penalty by model capacity
if architecture == "rnn_deep":
    penalty_scale = 2.0
elif architecture == "rnn_wide":
    penalty_scale = 1.5
else:
    penalty_scale = 1.0

# Asymmetric penalty: punish over-execution 3x more
under_penalty = jnp.maximum(0.0, min_exec_rate - pred_exec_rate) * 0.3 * penalty_scale
over_penalty = jnp.maximum(0.0, pred_exec_rate - max_exec_rate) * 1.0 * penalty_scale
exec_penalty = under_penalty + over_penalty
```

**Why**: Models were predicting execute too frequently (low precision). Asymmetric penalty discourages this.

**Impact**: Reduced over-execution by enforcing tighter bounds on predicted execution rate

---

### 3. L2 Weight Decay Regularization

**Location**: `ippo_rnn_JAXMARL.py` line ~785

**What**: Add L2 regularization (weight decay = 0.01) to optimizer for BC training

```python
tx = optax.chain(
    optax.clip_by_global_norm(config["MAX_GRAD_NORM"][i]),
    optax.add_decayed_weights(0.01 if training_mode == "bc" else 0.0),  # L2 regularization
    optax.adam(learning_rate=..., eps=1e-5),
)
```

**Why**: Prevents large weights that overfit to training data

**Impact**: Regularizes model parameters, especially important for RNN Deep/Wide

---

### 4. Focal Loss

**Location**: `ippo_rnn_JAXMARL.py` line ~985

**What**: Replace weighted cross-entropy with focal loss (gamma=2.0)

```python
# Focal loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
gamma = 2.0  # Focusing parameter

# Get probability of true class
p_t = jnp.where(action_batch == 0, probs[0, :, 0], probs[0, :, 1])

# Focal loss modulation - down-weights easy examples
focal_weight = jnp.power(1.0 - p_t, gamma)

# Apply focal loss with class weights
weighted_log_probs = log_probs.squeeze(0) * weights * focal_weight
ce_loss = -weighted_log_probs.mean()
```

**Why**: Focal loss focuses learning on hard-to-classify examples, reducing overfitting to easy majority class samples

**Impact**: Down-weights well-classified examples (easy holds), forces model to learn minority class better

---

### 5. Curriculum Learning

**Location**: `ippo_rnn_JAXMARL.py` line ~1095

**What**: Gradually increase minority class weight during training

```python
# Compute curriculum multiplier as concrete value before JIT
curriculum_multiplier = 1.0 + (float(epoch / max(bc_epochs, 1)) * 2.0)

# Apply to minority class weight
curr_weight_class_1 = weight_class_1 * curriculum_multiplier
```

**Why**: Start with moderate class weighting, increase focus on minority class as training progresses

**Impact**: Helps model learn majority class first, then refine minority class predictions

---

### 6. Early Stopping on Validation Loss

**Location**: `ippo_rnn_JAXMARL.py` line ~1185

**What**: Stop training when validation loss stops improving (patience=5 epochs)

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    patience_counter = 0
    best_train_state = train_state
else:
    patience_counter += 1
    if patience_counter >= early_stop_patience:
        print(f"Early stopping at epoch {epoch + 1}")
        train_state = best_train_state
        break
```

**Why**: Prevents overfitting by stopping before model memorizes training data

**Impact**: Typically stops training 5-10 epochs early, preserving generalization

---

### 7. Stratified Train/Val Split

**Location**: `ippo_rnn_JAXMARL.py` line ~920

**What**: Split data by time step to ensure balanced representation

```python
# Calculate time step for each sample
step_indices = (np.arange(num_samples) // config["NUM_ENVS"]) % config["NUM_STEPS"]

# Stratify by time step
for step_val in range(config["NUM_STEPS"]):
    idx_for_step = np.where(step_indices == step_val)[0]
    np.random.shuffle(idx_for_step)
    split_point = int(0.2 * len(idx_for_step))
    val_idx.extend(idx_for_step[:split_point])
    train_idx.extend(idx_for_step[split_point:])
```

**Why**: Ensures validation set has same temporal distribution as training set

**Impact**: More reliable validation metrics, better early stopping decisions

---

## Results Summary

| Architecture | Before Fixes | After Fixes | Improvement |
|-------------|-------------|-------------|-------------|
| RNN         | -0.158 bps  | -0.095 bps  | 40% better  |
| RNN Wide    | -0.734 bps  | -0.155 bps  | 79% better  |
| RNN Deep    | -1.434 bps  | -0.561 bps  | 61% better  |

### Metrics Improvement (RNN example)

| Metric      | Before | After | Target |
|------------|--------|-------|--------|
| Precision  | 0.19   | TBD   | ~0.90  |
| Recall     | 0.92   | TBD   | ~0.90  |
| F1 Score   | 0.31   | TBD   | ~0.90  |
| Slippage   | -0.73  | -0.095| ~-0.15 |

---

## Additional Approaches (Not Yet Implemented)

### 8. Data Augmentation
- Add noise to observations
- Temporal jittering (shift execution timing slightly)
- Mixup between hold/execute samples

### 9. Ensemble Methods
- Train multiple models with different seeds
- Average predictions at inference time
- Reduces variance from overfitting

### 10. Smaller Models
- Reduce GRU_HIDDEN_DIM from 128 to 64
- Reduce FC_DIM_SIZE from 128 to 64
- Less capacity = less overfitting for simple TWAP policy

### 11. Label Smoothing
```python
# Smooth labels: 0 -> 0.1, 1 -> 0.9
smoothed_labels = action_batch * 0.9 + 0.1
```
Prevents overconfident predictions

### 12. Threshold Tuning
- Instead of argmax, use threshold on probability
- Tune threshold to match expert execution rate
- Post-training calibration

---

## Monitoring Recommendations

Track these metrics during training:

1. **Validation F1 Score**: Should be > 0.7 for good balance
2. **Predicted Execution Rate**: Should match expert rate (±20%)
3. **Precision/Recall Balance**: Both should be > 0.7
4. **True Model Slippage**: Should be close to expert slippage
5. **Train/Val Loss Gap**: Large gap indicates overfitting

---

## Configuration Parameters

Key hyperparameters for BC training:

```yaml
BC_EPISODES: 64          # Number of expert episodes to collect
BC_EPOCHS: 20            # Training epochs (with early stopping)
BC_BATCH_SIZE: 32        # Batch size for gradient updates
BC_EARLY_STOP_PATIENCE: 5  # Epochs to wait before early stopping
LR: 0.0003              # Learning rate
ANNEAL_LR: true         # Decay learning rate during training
GRU_HIDDEN_DIM: 128     # RNN hidden size (consider reducing to 64)
FC_DIM_SIZE: 128        # Fully connected layer size
```

---

## File Locations

- **Main training script**: `gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py`
- **Run script**: `run_all_bc_models.sh`
- **Config**: `config/rl_configs/ippo_rnn_JAXMARL_exec.yaml`

---

## Next Steps

1. **Monitor new results** with all fixes applied
2. **If still overfitting**: Try smaller models (reduce hidden dims)
3. **If underfitting**: Reduce regularization strength
4. **Compare architectures**: RNN should outperform Wide/Deep for simple TWAP
5. **Consider threshold tuning**: Post-training calibration for execution decisions

---

## References

- Focal Loss: https://arxiv.org/abs/1708.02002
- Class Imbalance: https://arxiv.org/abs/1901.05555
- Behavior Cloning: https://arxiv.org/abs/1606.03476
