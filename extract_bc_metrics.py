#!/usr/bin/env python3
"""
Extract BC test metrics from W&B runs (from all runs, not just recent)
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Initialize W&B API
api = wandb.Api()

# Get all runs from the project
project_path = "eitansomething-n-a/v4_Exec_FixedQuantsComplex_Ablations"
runs_all = list(api.runs(project_path))
print(f"Total runs in project: {len(runs_all)}")

# Extract metrics from runs with BC metrics
results = []

for run in runs_all:
    # Look for BC metrics in summary
    has_bc_metrics = any('bc/' in k for k in run.summary.keys())
    
    if not has_bc_metrics:
        continue
    
    # Extract architecture and policy from run tags or notes
    tags = run.tags if hasattr(run, 'tags') else []
    notes = run.notes if hasattr(run, 'notes') else ""
    
    # Try to extract from  run name pattern
    run_name = run.name
    
    # Parse run name format
    architecture = None
    policy = None
    
    # Check config for architecture and policy
    if hasattr(run, 'config'):
        architecture = run.config.get('ARCHITECTURE')
        policy = run.config.get('EXPERT_POLICY')
    
    # If not in config, try to infer from summary metrics
    if not architecture or not policy:
        # Look at metric keys to infer config
        metric_keys = [k for k in run.summary.keys() if 'bc/' in k]
        if metric_keys:
            # Sample key might be "bc/agent_EXE_val_accuracy"
            sample_key = metric_keys[0]
            # Extract policy from run name or other metadata
            if 'vwap' in run_name.lower() or 'vwap' in str(notes).lower():
                policy = 'vwap'
            elif 'twap' in run_name.lower() or 'twap' in str(notes).lower():
                policy = 'twap'
    
    # Extract all BC metrics
    bc_metrics = {}
    for key, value in run.summary.items():
        if 'bc/' in key and isinstance(value, (int, float)) and not np.isnan(value):
            # Normalize key names
            if 'val_accuracy' in key:
                if 'val_accuracy' not in bc_metrics:
                    bc_metrics['val_accuracy'] = value
            elif 'val_f1' in key:
                if 'val_f1' not in bc_metrics:
                    bc_metrics['val_f1'] = value
            elif 'val_precision' in key:
                if 'val_precision' not in bc_metrics:
                    bc_metrics['val_precision'] = value
            elif 'val_recall' in key:
                if 'val_recall' not in bc_metrics:
                    bc_metrics['val_recall'] = value
    
    # Only include runs with at least accuracy metric
    if 'val_accuracy' in bc_metrics:
        results.append({
            'run_name': run_name,
            'run_id': run.id,
            'architecture': architecture or 'unknown',
            'policy': policy or 'unknown',
            'val_accuracy': bc_metrics.get('val_accuracy'),
            'val_f1': bc_metrics.get('val_f1', np.nan),
            'val_precision': bc_metrics.get('val_precision', np.nan),
            'val_recall': bc_metrics.get('val_recall', np.nan),
        })

df = pd.DataFrame(results)

if len(df) == 0:
    print("\nNo BC test metrics found!")
else:
    print(f"\n✓ Found {len(df)} runs with BC test metrics\n")
    
    # Display full table
    print("="*110)
    print(f"{'Run Name':<25} {'Architecture':<15} {'Policy':<10} {'Val Accuracy':<15} {'Val F1':<12} {'Precision':<12} {'Recall':<12}")
    print("="*110)
    
    df_sorted = df.sort_values(['policy', 'architecture'])
    for _, row in df_sorted.iterrows():
        acc_str = f"{row['val_accuracy']:.4f}" if pd.notna(row['val_accuracy']) else "N/A"
        f1_str = f"{row['val_f1']:.4f}" if pd.notna(row['val_f1']) else "N/A"
        prec_str = f"{row['val_precision']:.4f}" if pd.notna(row['val_precision']) else "N/A"
        rec_str = f"{row['val_recall']:.4f}" if pd.notna(row['val_recall']) else "N/A"
        
        print(f"{row['run_name']:<25} {str(row['architecture']):<15} {str(row['policy']):<10} " +
              f"{acc_str:<15} {f1_str:<12} {prec_str:<12} {rec_str:<12}")
    
    print("="*110)
    
    # Save to CSV
    df.to_csv("bc_test_metrics.csv", index=False)
    print(f"\n✓ Metrics saved to bc_test_metrics.csv")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("BC Training: Test Metrics Comparison", fontsize=14, fontweight='bold')
    
    # Prepare data for plotting
    valid_df = df[df['architecture'] != 'unknown'].copy()
    
    if len(valid_df) > 0:
        # Plot 1: Accuracy by Architecture (grouped by Policy)
        policies = sorted(valid_df['policy'].unique())
        architectures = ['rnn', 'rnn_wide', 'rnn_deep', 'transformer']
        
        for policy in policies:
            policy_data = valid_df[valid_df['policy'] == policy]
            arch_accs = []
            for arch in architectures:
                arch_subset = policy_data[policy_data['architecture'] == arch]
                if len(arch_subset) > 0:
                    arch_accs.append(arch_subset['val_accuracy'].mean())
                else:
                    arch_accs.append(np.nan)
            axes[0, 0].plot(range(len(architectures)), arch_accs, marker="o", label=policy.upper(), linewidth=2)
        
        axes[0, 0].set_xticks(range(len(architectures)))
        axes[0, 0].set_xticklabels(architectures, rotation=45, ha='right')
        axes[0, 0].set_ylabel("Validation Accuracy", fontweight='bold')
        axes[0, 0].set_title("Accuracy by Architecture")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0.5, 1.0])
        
        # Plot 2: F1 by Architecture
        for policy in policies:
            policy_data = valid_df[valid_df['policy'] == policy]
            arch_f1s = []
            for arch in architectures:
                arch_subset = policy_data[policy_data['architecture'] == arch]
                if len(arch_subset) > 0:
                    f1_vals = arch_subset['val_f1'].dropna()
                    if len(f1_vals) > 0:
                        arch_f1s.append(f1_vals.mean())
                    else:
                        arch_f1s.append(np.nan)
                else:
                    arch_f1s.append(np.nan)
            axes[0, 1].plot(range(len(architectures)), arch_f1s, marker="s", label=policy.upper(), linewidth=2)
        
        axes[0, 1].set_xticks(range(len(architectures)))
        axes[0, 1].set_xticklabels(architectures, rotation=45, ha='right')
        axes[0, 1].set_ylabel("Validation F1 Score", fontweight='bold')
        axes[0, 1].set_title("F1 Score by Architecture")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0.5, 1.0])
        
        # Plot 3: Precision vs Recall by Policy
        policy_stats = []
        for policy in policies:
            policy_data = valid_df[valid_df['policy'] == policy]
            avg_prec = policy_data['val_precision'].mean()
            avg_recall = policy_data['val_recall'].mean()
            policy_stats.append((policy, avg_prec, avg_recall))
        
        x_pos = np.arange(len(policy_stats))
        width = 0.35
        
        precisions = [x[1] for x in policy_stats]
        recalls = [x[2] for x in policy_stats]
        
        axes[1, 0].bar(x_pos - width/2, precisions, width, label='Precision', alpha=0.8)
        axes[1, 0].bar(x_pos + width/2, recalls, width, label='Recall', alpha=0.8)
        
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels([p.upper() for p in policies])
        axes[1, 0].set_ylabel("Score", fontweight='bold')
        axes[1, 0].set_title("Precision vs Recall by Policy")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].set_ylim([0, 1])
        
        # Plot 4: Summary statistics
        axes[1, 1].axis('off')
        summary_text = "Test Metrics Summary\n" + "=" * 45 + "\n"
        
        for policy in policies:
            policy_data = valid_df[valid_df['policy'] == policy]
            avg_acc = policy_data['val_accuracy'].mean()
            max_acc = policy_data['val_accuracy'].max()
            avg_f1 = policy_data['val_f1'].mean()
            best_arch = policy_data.loc[policy_data['val_accuracy'].idxmax(), 'architecture']
            
            summary_text += f"\n{policy.upper()}:\n"
            summary_text += f"  Avg Accuracy: {avg_acc:.4f}\n"
            summary_text += f"  Max Accuracy: {max_acc:.4f} ({best_arch})\n"
            summary_text += f"  Avg F1: {avg_f1:.4f}\n"
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                       fontfamily='monospace', fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig("bc_test_metrics_chart.png", dpi=150, bbox_inches="tight")
        print("✓ Chart saved to bc_test_metrics_chart.png\n")
        
        # Print summary
        print("Test Metrics Summary:")
        print("=" * 50)
        for policy in policies:
            policy_data = valid_df[valid_df['policy'] == policy]
            print(f"\n{policy.upper()} Policy:")
            print(f"  Average Accuracy:  {policy_data['val_accuracy'].mean():.4f}")
            f1_mean = policy_data['val_f1'].mean()
            if not np.isnan(f1_mean):
                print(f"  Average F1 Score:  {f1_mean:.4f}")
            prec_mean = policy_data['val_precision'].mean()
            if not np.isnan(prec_mean):
                print(f"  Average Precision: {prec_mean:.4f}")
            rec_mean = policy_data['val_recall'].mean()
            if not np.isnan(rec_mean):
                print(f"  Average Recall:    {rec_mean:.4f}")
            
            best_idx = policy_data['val_accuracy'].idxmax()
            best_arch = policy_data.loc[best_idx, 'architecture']
            best_acc = policy_data.loc[best_idx, 'val_accuracy']
            print(f"  Best: {best_arch} ({best_acc:.4f})")
    else:
        print("Could not create plots - no runs with known architectures")
