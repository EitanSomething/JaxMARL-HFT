#!/usr/bin/env python3
"""
Extract BC training slippage metrics from W&B runs
Slippage is measured in basis points (bps)
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Initialize W&B API
api = wandb.Api()

# Get all runs from project
project_path = "eitansomething-n-a/v4_Exec_FixedQuantsComplex_Ablations"
runs_all = list(api.runs(project_path))
print(f"Total runs in project: {len(runs_all)}")

# Extract slippage metrics
results = []

for run in runs_all:
    # Look for BC slippage metrics in summary
    has_bc_metrics = any('bc/' in k for k in run.summary.keys())
    
    if not has_bc_metrics:
        continue
    
    # Try to get architecture and policy from config
    architecture = run.config.get('ARCHITECTURE') if hasattr(run, 'config') else None
    policy = run.config.get('EXPERT_POLICY') if hasattr(run, 'config') else None
    
    # Extract slippage metrics
    slippage_value = None
    for key, value in run.summary.items():
        if 'slippage' in key.lower() and 'bc/' in key and isinstance(value, (int, float)) and not np.isnan(value):
            slippage_value = value
            break
    
    # If we found slippage and have architecture/policy, add to results
    if slippage_value is not None and architecture and policy:
        results.append({
            'run_name': run.name,
            'run_id': run.id,
            'architecture': architecture,
            'policy': policy,
            'slippage_bps': slippage_value,
        })

df = pd.DataFrame(results)

if len(df) == 0:
    print("\n❌ No BC slippage metrics found in runs")
    print("\nScanning runs for available slippage-related keys...")
    
    # Check what keys are available
    for run in runs_all[-10:]:
        slippage_keys = [k for k in run.summary.keys() if 'slippage' in k.lower()]
        if slippage_keys:
            print(f"\nRun {run.name}:")
            for key in slippage_keys[:3]:
                print(f"  {key}: {run.summary[key]}")
else:
    print(f"\n✓ Found {len(df)} runs with slippage metrics\n")
    
    # Display table sorted by policy then architecture
    print("="*90)
    print(f"{'Run Name':<30} {'Architecture':<15} {'Policy':<10} {'Slippage (bps)':<15}")
    print("="*90)
    
    df_sorted = df.sort_values(['policy', 'architecture'])
    for _, row in df_sorted.iterrows():
        slip_str = f"{row['slippage_bps']:.4f}"
        print(f"{row['run_name']:<30} {str(row['architecture']):<15} {str(row['policy']):<10} {slip_str:<15}")
    
    print("="*90)
    
    # Save to CSV
    df.to_csv("bc_slippage_metrics.csv", index=False)
    print(f"\n✓ Slippage metrics saved to bc_slippage_metrics.csv")
    
    # Summary statistics
    print("\nSlippage Summary (basis points):")
    print("="*50)
    
    for policy in sorted(df['policy'].unique()):
        policy_data = df[df['policy'] == policy]
        print(f"\n{policy.upper()}:")
        print(f"  Average Slippage: {policy_data['slippage_bps'].mean():>10.4f} bps")
        print(f"  Median Slippage:  {policy_data['slippage_bps'].median():>10.4f} bps")
        print(f"  Min Slippage:     {policy_data['slippage_bps'].min():>10.4f} bps (best)")
        print(f"  Max Slippage:     {policy_data['slippage_bps'].max():>10.4f} bps (worst)")
        
        best_idx = policy_data['slippage_bps'].idxmin()
        best_arch = policy_data.loc[best_idx, 'architecture']
        best_slip = policy_data.loc[best_idx, 'slippage_bps']
        print(f"  Best: {best_arch} ({best_slip:.4f} bps)")
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("BC Training: Slippage Comparison (in basis points)", fontsize=14, fontweight='bold')
    
    # Plot 1: Slippage by Architecture (grouped by Policy)
    architectures = ['rnn', 'rnn_wide', 'rnn_deep', 'transformer']
    policies = sorted(df['policy'].unique())
    
    for policy in policies:
        policy_data = df[df['policy'] == policy]
        arch_slips = []
        for arch in architectures:
            arch_subset = policy_data[policy_data['architecture'] == arch]
            if len(arch_subset) > 0:
                arch_slips.append(arch_subset['slippage_bps'].mean())
            else:
                arch_slips.append(np.nan)
        axes[0].plot(range(len(architectures)), arch_slips, marker="o", label=policy.upper(), linewidth=2)
    
    axes[0].set_xticks(range(len(architectures)))
    axes[0].set_xticklabels(architectures, rotation=45, ha='right')
    axes[0].set_ylabel("Slippage (basis points)", fontweight='bold')
    axes[0].set_title("Slippage by Architecture")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero slippage')
    
    # Plot 2: Box plot of slippage by policy
    policy_slips = [df[df['policy'] == p]['slippage_bps'].values for p in policies]
    bp = axes[1].boxplot(policy_slips, labels=[p.upper() for p in policies], patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    axes[1].set_ylabel("Slippage (basis points)", fontweight='bold')
    axes[1].set_title("Slippage Distribution by Policy")
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Add mean markers
    for i, policy in enumerate(policies):
        policy_data = df[df['policy'] == policy]
        mean_slip = policy_data['slippage_bps'].mean()
        axes[1].scatter([i+1], [mean_slip], color='red', s=100, marker='D', zorder=3, label='Mean' if i == 0 else '')
    
    plt.tight_layout()
    plt.savefig("bc_slippage_comparison.png", dpi=150, bbox_inches="tight")
    print("\n✓ Chart saved to bc_slippage_comparison.png\n")
    
    # Print comparison with accuracy
    print("\nComparison: Accuracy vs Slippage")
    print("="*70)
    print(f"{'Policy':<10} {'Metric':<20} {'Avg Value':<15} {'Range':<20}")
    print("="*70)
    
    # Read accuracy metrics
    try:
        df_acc = pd.read_csv("bc_test_metrics.csv")
        for policy in sorted(df['policy'].unique()):
            policy_acc = df_acc[df_acc['policy'] == policy]
            policy_slip = df[df['policy'] == policy]
            
            print(f"{policy.upper():<10} {'Accuracy':<20} {policy_acc['val_accuracy'].mean():<15.4f} " +
                  f"{policy_acc['val_accuracy'].min():.4f}-{policy_acc['val_accuracy'].max():.4f}")
            print(f"{'':<10} {'Slippage (bps)':<20} {policy_slip['slippage_bps'].mean():<15.4f} " +
                  f"{policy_slip['slippage_bps'].min():.4f}-{policy_slip['slippage_bps'].max():.4f}")
            print()
    except FileNotFoundError:
        print("(Accuracy metrics file not found)")
