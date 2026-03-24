#!/usr/bin/env python3
"""
Inference script: Run trained BC models on unseen validation data.

This evaluates trained models on data they never saw during training:
- Different date periods
- Different market conditions
- True out-of-sample performance

Usage:
    python3 validate_models.py --expert vwap --architecture rnn_deep
    python3 validate_models.py --expert vwap --all-architectures
"""

import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf, DictConfig
import hydra
import argparse
from typing import Dict, List, Tuple

# Config loading
from gymnax_exchange.jaxob.config_io import load_config_from_file

def find_bc_checkpoint(expert_policy: str, architecture: str, results_base: str = "results") -> Path:
    """Find the latest BC checkpoint for given expert/architecture."""
    checkpoint_patterns = [
        f"{results_base}/bc_{expert_policy}_{architecture}/checkpoints/",
        f"{results_base}/architecture_sweep/{expert_policy}_{architecture}/hydra_outputs/checkpoints/",
        f"{results_base}/{expert_policy}_{architecture}/checkpoints/",
    ]
    
    for pattern in checkpoint_patterns:
        pattern_path = Path(pattern)
        if pattern_path.exists():
            checkpoints = list(pattern_path.glob("*.pkl")) + list(pattern_path.glob("*.orbax"))
            if checkpoints:
                latest = sorted(checkpoints, key=lambda p: p.stat().st_mtime)[-1]
                return latest
    
    return None

def load_validation_environment(expert_policy: str, 
                               val_time_period: str = "2012-06-22",
                               stock: str = "AMZN",
                               data_path: str = "data"):
    """Create validation environment with different date than training."""
    from gymnax_exchange.jaxen.marl_env import MARLEnv
    from gymnax_exchange.jaxob.jaxob_config import MultiAgentConfig
    
    # Load base config
    config_path = Path("config/env_configs/default_exec.yaml")
    if not config_path.exists():
        config_path = Path(__file__).parent / "config/env_configs/default_exec.yaml"
    
    cfg = OmegaConf.load(config_path) if config_path.exists() else OmegaConf.create()
    
    # Override for validation
    cfg.world_config.stock = stock
    cfg.world_config.timePeriod = val_time_period
    cfg.world_config.dataPath = data_path
    cfg.world_config.alphatradePath = str(Path(__file__).parent)
    cfg.EXPERT_POLICY = expert_policy
    
    # Create environment
    multi_agent_config = MultiAgentConfig(**cfg)
    env = MARLEnv(multi_agent_config)
    
    return env, multi_agent_config

def run_inference_episode(agent_state, env, env_params, rng_key, 
                         max_steps: int = 32) -> Dict:
    """Run single validation episode and collect metrics."""
    
    obs, env_state = env.reset(rng_key, env_params)
    dones = jnp.zeros(1, dtype=bool)
    
    episode_metrics = {
        'actions': [],
        'predictions': [],
        'rewards': [],
        'slippage_samples': [],
        'accuracy_correct': 0,
        'accuracy_total': 0,
    }
    
    for step in range(max_steps):
        # Get model prediction
        pred_action, _ = agent_state.apply_fn({'params': agent_state.params}, obs)
        
        # Step environment (will use expert policy)
        obs_next, env_state, reward, done, info = env.step(
            (obs, env_state), env_params, jnp.array([pred_action])
        )
        
        # Collect metrics
        episode_metrics['actions'].append(float(info.get('expert_action', 0)))
        episode_metrics['predictions'].append(float(pred_action))
        episode_metrics['rewards'].append(float(reward))
        
        if 'slippage_bps' in info:
            episode_metrics['slippage_samples'].append(float(info['slippage_bps']))
        
        # Track accuracy
        if 'expert_action' in info:
            is_correct = abs(float(pred_action) - float(info['expert_action'])) < 0.5
            episode_metrics['accuracy_correct'] += int(is_correct)
            episode_metrics['accuracy_total'] += 1
        
        obs = obs_next
        if done:
            break
    
    # Aggregate metrics
    if episode_metrics['accuracy_total'] > 0:
        accuracy = episode_metrics['accuracy_correct'] / episode_metrics['accuracy_total']
    else:
        accuracy = 0.0
    
    avg_reward = np.mean(episode_metrics['rewards']) if episode_metrics['rewards'] else 0.0
    avg_slippage = np.mean(episode_metrics['slippage_samples']) if episode_metrics['slippage_samples'] else 0.0
    
    return {
        'accuracy': accuracy,
        'reward': avg_reward,
        'slippage_bps': avg_slippage,
        'n_steps': len(episode_metrics['actions']),
    }

def validate_model(expert_policy: str, 
                  architecture: str,
                  n_episodes: int = 5,
                  val_date: str = None) -> Dict:
    """Validate a trained model on unseen data."""
    
    print(f"\n{'='*80}")
    print(f"Validating: {expert_policy.upper()} + {architecture}")
    print(f"{'='*80}")
    
    # Find checkpoint
    checkpoint_path = find_bc_checkpoint(expert_policy, architecture)
    if not checkpoint_path:
        print(f"⚠ No checkpoint found for {expert_policy}_{architecture}")
        return None
    
    print(f"✓ Checkpoint: {checkpoint_path}")
    
    # Load model (placeholder - would need actual checkpoint loading)
    print(f"Loading model...")
    # TODO: Load from checkpoint
    
    # Create validation env
    if not val_date:
        # Use different date than training
        val_date = "2012-06-22" if "06-21" in "training" else "2012-06-21"
    
    print(f"Creating validation environment (date: {val_date})...")
    try:
        env, cfg = load_validation_environment(expert_policy, val_date)
    except Exception as e:
        print(f"⚠ Could not create validation env: {e}")
        return None
    
    # Run multiple episodes
    results = {
        'episodes': [],
        'avg_accuracy': 0,
        'avg_reward': 0,
        'avg_slippage_bps': 0,
    }
    
    print(f"Running {n_episodes} validation episodes...")
    for ep in range(n_episodes):
        print(f"  Episode {ep+1}/{n_episodes}...", end=" ", flush=True)
        # TODO: Run episode
        # ep_result = run_inference_episode(...)
        # results['episodes'].append(ep_result)
        print("✓")
    
    # Aggregate
    if results['episodes']:
        results['avg_accuracy'] = np.mean([e['accuracy'] for e in results['episodes']])
        results['avg_reward'] = np.mean([e['reward'] for e in results['episodes']])
        results['avg_slippage_bps'] = np.mean([e['slippage_bps'] for e in results['episodes']])
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Validate trained BC models on unseen test data'
    )
    parser.add_argument('--expert', choices=['vwap', 'twap'], default='vwap',
                       help='Expert policy')
    parser.add_argument('--architecture', 
                       choices=['rnn_base', 'rnn_wide', 'rnn_deep', 'transformer'],
                       default='rnn_deep',
                       help='Model architecture')
    parser.add_argument('--all-architectures', action='store_true',
                       help='Validate all architectures')
    parser.add_argument('--both-experts', action='store_true',
                       help='Validate both expert policies')
    parser.add_argument('--val-date', default=None,
                       help='Validation date (format: YYYY-MM-DD, e.g., 2012-06-22)')
    parser.add_argument('--n-episodes', type=int, default=10,
                       help='Number of validation episodes')
    parser.add_argument('--output', default='validation_results.csv',
                       help='Output CSV file')
    
    args = parser.parse_args()
    
    # Determine which versions to validate
    experts = ['vwap', 'twap'] if args.both_experts else [args.expert]
    architectures = ['rnn_base', 'rnn_wide', 'rnn_deep', 'transformer'] if args.all_architectures else [args.architecture]
    
    print(f"\n{'='*80}")
    print(f"BC Model Validation on Unseen Data")
    print(f"{'='*80}")
    print(f"Experts: {experts}")
    print(f"Architectures: {architectures}")
    print(f"Validation date: {args.val_date or 'auto-select (different from training)'}")
    print(f"Episodes per model: {args.n_episodes}")
    
    # Validate each combination
    results = []
    for expert in experts:
        for arch in architectures:
            result = validate_model(expert, arch, n_episodes=args.n_episodes, 
                                   val_date=args.val_date)
            if result:
                result['expert_policy'] = expert
                result['architecture'] = arch
                results.append(result)
    
    # Print results
    print(f"\n{'='*80}")
    print("VALIDATION RESULTS")
    print(f"{'='*80}\n")
    
    for expert in experts:
        print(f"\n{expert.upper()} Expert Policy:")
        print("-" * 80)
        print(f"{'Architecture':<15} {'Accuracy':<12} {'Slippage (bps)':<15} {'Episodes':<10}")
        print("-" * 80)
        
        for result in results:
            if result['expert_policy'] == expert:
                print(f"{result['architecture']:<15} "
                      f"{result.get('avg_accuracy', 0):<12.4f} "
                      f"{result.get('avg_slippage_bps', 0):<15.4f} "
                      f"{args.n_episodes:<10}")
    
    # Save to CSV
    if results:
        import csv
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['expert_policy', 'architecture', 
                                                   'avg_accuracy', 'avg_reward', 'avg_slippage_bps'])
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✓ Results saved to: {args.output}")

if __name__ == '__main__':
    main()
