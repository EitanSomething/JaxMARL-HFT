#!/usr/bin/env python3
"""
Diagnostic tool to check what metrics are available in a W&B run.
Usage: python3 check_run_metrics.py --run rnn_base_vwap
"""

import argparse
import wandb
from pathlib import Path
import json

def check_wandb_metrics(entity: str, project: str, run_name: str = None):
    """Check available metrics in W&B runs."""
    api = wandb.Api()
    project_path = f"{entity}/{project}"
    
    print(f"Checking {project_path}...")
    runs = api.runs(project_path)
    
    if run_name:
        runs = [r for r in runs if run_name.lower() in r.name.lower()]
    
    if not runs:
        print(f"No runs found matching: {run_name}")
        return
    
    for run in runs:
        print(f"\n{'='*80}")
        print(f"Run: {run.name}")
        print(f"Status: {run.state}")
        print(f"{'='*80}")
        
        print("\nConfig:")
        for key in ['EXPERT_POLICY', 'ARCHITECTURE', 'TRAINING_MODE', 'BC_EPOCHS']:
            val = run.config.get(key)
            if val is not None:
                print(f"  {key}: {val}")
        
        print("\nMetrics Summary (available):")
        summary = run.summary
        
        # Group metrics by prefix
        metrics_by_type = {}
        for key, val in summary.items():
            if isinstance(val, (int, float)):
                prefix = key.split('/')[0] if '/' in key else 'other'
                if prefix not in metrics_by_type:
                    metrics_by_type[prefix] = []
                metrics_by_type[prefix].append((key, val))
        
        for prefix in sorted(metrics_by_type.keys()):
            print(f"\n  {prefix}:")
            for key, val in sorted(metrics_by_type[prefix]):
                if isinstance(val, float):
                    print(f"    {key}: {val:.6f}")
                else:
                    print(f"    {key}: {val}")
        
        # Highlight key metrics
        print("\nKey Metrics:")
        key_metrics = [
            'bc/agent_EXE/val_accuracy',
            'bc/agent_EXE/val_f1',
            'bc/agent_EXE/val_precision',
            'bc/agent_EXE/val_recall',
            'bc/agent_EXE/price_slippage_bps',
            'bc/agent_EXE/training_time_seconds',
        ]
        
        found_any = False
        for metric in key_metrics:
            val = summary.get(metric)
            if val is not None:
                print(f"  ✓ {metric}: {val}")
                found_any = True
        
        if not found_any:
            print("  ⚠ No key metrics found!")
            print("\n  All available keys:")
            for key in sorted(summary.keys()):
                if isinstance(summary[key], (int, float)):
                    print(f"    - {key}")

def check_local_logs(results_dir: str):
    """Check local log files for metric patterns."""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    print(f"\nChecking local logs in: {results_dir}")
    print("="*80)
    
    for run_dir in sorted(results_path.glob("*")):
        if not run_dir.is_dir():
            continue
        
        log_file = run_dir / "train.log"
        if not log_file.exists():
            continue
        
        print(f"\n{run_dir.name}:")
        
        with open(log_file, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            
            # Look for metric patterns
            metric_patterns = [
                'accuracy', 'val_accuracy',
                'f1', 'val_f1',
                'precision', 'slippage',
                'loss', 'epoch'
            ]
            
            found_metrics = []
            for line in lines:
                for pattern in metric_patterns:
                    if pattern in line.lower():
                        found_metrics.append(line.strip())
                        break
            
            if found_metrics:
                print(f"  Found {len(found_metrics)} metric lines:")
                for line in found_metrics[-5:]:  # Show last 5
                    if len(line) > 150:
                        print(f"    {line[:150]}...")
                    else:
                        print(f"    {line}")
            else:
                print("  No metric patterns found in log")

def main():
    parser = argparse.ArgumentParser(description='Check available metrics in W&B runs')
    parser.add_argument('--entity', default='eitansomething-n-a', help='W&B entity')
    parser.add_argument('--project', default='exec_env_bc', help='W&B project')
    parser.add_argument('--run', default=None, help='Specific run name (partial match)')
    parser.add_argument('--logs', default='results/architecture_sweep', help='Check local logs')
    
    args = parser.parse_args()
    
    # Check W&B
    try:
        check_wandb_metrics(args.entity, args.project, args.run)
    except Exception as e:
        print(f"Error checking W&B: {e}")
        print("Make sure you've run: wandb login")
    
    # Check local logs
    if Path(args.logs).exists():
        check_local_logs(args.logs)

if __name__ == '__main__':
    main()
