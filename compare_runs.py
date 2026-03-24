#!/usr/bin/env python3
"""
Compare architecture runs from W&B and generate comparison table.
Usage: python3 compare_runs.py --entity eitansomething-n-a --project exec_env_bc
"""

import argparse
import pandas as pd
from pathlib import Path
import wandb
from typing import List, Dict
import sys

def get_wandb_runs(entity: str, project: str, filters: Dict = None, debug: bool = False) -> List[Dict]:
    """Fetch runs from W&B matching filters."""
    try:
        api = wandb.Api()
    except Exception as e:
        print(f"Error initializing W&B API: {e}")
        print("Make sure you've run: wandb login")
        sys.exit(1)
    
    # Build project path
    project_path = f"{entity}/{project}"
    
    if debug:
        print(f"Fetching runs from: {project_path}")
    
    try:
        runs = api.runs(project_path)
    except Exception as e:
        print(f"Error fetching runs: {e}")
        print(f"Project may not exist: {project_path}")
        return []
    
    results = []
    for i, run in enumerate(runs):
        if debug:
            print(f"  [{i+1}] {run.name} ({run.state}) - ", end="", flush=True)
        
        if filters:
            skip = False
            for key, value in filters.items():
                if run.config.get(key) != value:
                    skip = True
                    break
            if skip:
                if debug:
                    print(f"(filtered out)")
                continue
        
        # Extract summary metrics
        summary = run.summary
        config = run.config
        
        # Try multiple metric name variations
        accuracy = (
            summary.get('bc/agent_EXE/val_accuracy') or
            summary.get('val_accuracy') or
            summary.get('accuracy') or
            None
        )
        f1 = (
            summary.get('bc/agent_EXE/val_f1') or
            summary.get('val_f1') or
            summary.get('f1') or
            None
        )
        precision = (
            summary.get('bc/agent_EXE/val_precision') or
            summary.get('val_precision') or
            summary.get('precision') or
            None
        )
        recall = (
            summary.get('bc/agent_EXE/val_recall') or
            summary.get('val_recall') or
            summary.get('recall') or
            None
        )
        slippage_bps = (
            summary.get('bc/agent_EXE/price_slippage_bps') or
            summary.get('price_slippage_bps') or
            summary.get('slippage_bps') or
            None
        )
        
        result = {
            'run_id': run.id,
            'run_name': run.name,
            'status': run.state,
            'expert_policy': config.get('EXPERT_POLICY', 'unknown'),
            'architecture': config.get('ARCHITECTURE', 'unknown'),
            'training_mode': config.get('TRAINING_MODE', 'unknown'),
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'slippage_bps': slippage_bps,
            'epoch': config.get('BC_EPOCHS', None),
            'batch_size': config.get('BC_BATCH_SIZE', None),
        }
        results.append(result)
        
        if debug:
            has_metrics = any([accuracy, f1, precision, recall, slippage_bps])
            print(f"({len([x for x in [accuracy,f1,precision,recall,slippage_bps] if x])} metrics)")
    
    return results

def create_comparison_table(runs: List[Dict]) -> pd.DataFrame:
    """Create comparison DataFrame."""
    df = pd.DataFrame(runs)
    
    # Sort by expert policy and architecture
    df = df.sort_values(['expert_policy', 'architecture'])
    
    return df

def print_comparison(df: pd.DataFrame):
    """Print formatted comparison table."""
    print("\n" + "="*100)
    print("ARCHITECTURE COMPARISON - W&B Results")
    print("="*100)
    
    for expert in sorted(df['expert_policy'].unique()):
        subset = df[df['expert_policy'] == expert]
        print(f"\n{expert.upper()} Expert Policy:")
        print("-"*100)
        
        display_cols = ['architecture', 'accuracy', 'f1', 'precision', 'slippage_bps']
        subset_display = subset[display_cols].copy()
        subset_display.columns = ['Architecture', 'Accuracy', 'F1', 'Precision', 'Slippage (bps)']
        
        print(subset_display.to_string(index=False))
    
    print("\n" + "="*100)

def main():
    parser = argparse.ArgumentParser(description='Compare W&B runs across architectures')
    parser.add_argument('--entity', default='eitansomething-n-a', help='W&B entity')
    parser.add_argument('--project', default='exec_env_bc', help='W&B project')
    parser.add_argument('--output', default='architecture_comparison.csv', help='Output CSV file')
    parser.add_argument('--mode', default='bc', help='Training mode filter')
    parser.add_argument('--debug', action='store_true', help='Show debug info')
    
    args = parser.parse_args()
    
    print(f"Connecting to W&B: {args.entity}/{args.project}")
    print()
    
    # Fetch runs
    filters = {'TRAINING_MODE': args.mode} if args.mode else None
    runs = get_wandb_runs(args.entity, args.project, filters, debug=args.debug)
    
    if not runs:
        print(f"✗ No runs found for {args.mode} mode in {args.entity}/{args.project}")
        print("\nTroubleshooting:")
        print("  1. Verify project exists in W&B dashboard")
        print("  2. Check entity name: wandb.ai/<entity>/projects")
        print("  3. Ensure runs are in 'bc' training mode")
        print("  4. Try: python3 compare_runs.py --debug --mode=''")
        return
    
    print(f"✓ Found {len(runs)} runs\n")
    
    # Create comparison
    df = create_comparison_table(runs)
    
    # Print table
    print_comparison(df)
    
    # Save to CSV
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\n✓ Saved to: {args.output}\n")
    
    # Print summary statistics
    print("="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    
    for expert in sorted(df['expert_policy'].unique()):
        subset = df[df['expert_policy'] == expert][['architecture', 'accuracy', 'f1', 'slippage_bps']]
        print(f"\n{expert.upper()}:")
        print(subset.groupby('architecture')[['accuracy', 'f1', 'slippage_bps']].describe().round(4))
    
    # Check for missing data
    missing = df[df['accuracy'].isna()]
    if len(missing) > 0:
        print("\n" + "="*100)
        print("⚠ WARNING: Missing Metrics")
        print("="*100)
        print(f"\n{len(missing)} runs have no metrics:")
        for _, row in missing.iterrows():
            print(f"  - {row['run_name']} ({row['expert_policy']}/{row['architecture']})")
        print("\nPossible causes:")
        print("  1. Runs still training (wait for completion)")
        print("  2. Metrics not logged to W&B (check training code)")
        print("  3. Training failed (check run logs in W&B dashboard)")

if __name__ == '__main__':
    main()
