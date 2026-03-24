#!/usr/bin/env python3
"""
Extract metrics from local run logs and generate comparison table.
This is the fallback when W&B API is not available.
"""

import re
import csv
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

def extract_metrics_from_log(log_file: Path) -> Dict[str, float]:
    """Extract BC metrics from train.log file."""
    metrics = {}
    
    if not log_file.exists():
        return metrics
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            
            # Parse wandb metric lines: "wandb:      bc/agent_EXE/val_accuracy 0.61844"
            for line in lines:
                if 'wandb:' not in line:
                    continue
                
                # Extract metric name and value
                # Format: "wandb:      <metric_name> <value>"
                match = re.search(r'wandb:\s+([\w/]+)\s+([\d.-]+)', line)
                if not match:
                    continue
                
                metric_name = match.group(1)
                value = float(match.group(2))
                
                # Map metric names to simple keys
                if 'val_accuracy' in metric_name:
                    metrics['accuracy'] = value
                elif 'val_f1' in metric_name:
                    metrics['f1'] = value
                elif 'val_precision' in metric_name:
                    metrics['precision'] = value
                elif 'val_recall' in metric_name:
                    metrics['recall'] = value
                elif 'slippage_bps' in metric_name or 'price_slippage' in metric_name:
                    metrics['slippage_bps'] = value
    
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
    
    return metrics

def generate_comparison(results_dir: str = "results/architecture_sweep") -> List[Dict]:
    """Extract metrics from all runs and return results."""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return []
    
    results = []
    
    for run_dir in sorted(results_path.glob("*")):
        if not run_dir.is_dir():
            continue
        
        run_name = run_dir.name
        parts = run_name.rsplit('_', 1)  # Split from right to handle underscores in names
        if len(parts) != 2:
            continue
        
        expert_policy, architecture = parts
        log_file = run_dir / "train.log"
        
        metrics = extract_metrics_from_log(log_file)
        
        result = {
            'expert_policy': expert_policy,
            'architecture': architecture,
            'run_name': run_name,
            'accuracy': metrics.get('accuracy'),
            'f1': metrics.get('f1'),
            'precision': metrics.get('precision'),
            'recall': metrics.get('recall'),
            'slippage_bps': metrics.get('slippage_bps'),
        }
        results.append(result)
    
    return results

def print_results(results: List[Dict]):
    """Print formatted comparison table."""
    if not results:
        print("No results found!")
        return
    
    print("\n" + "="*120)
    print("ARCHITECTURE COMPARISON RESULTS")
    print("="*120)
    
    # Group by expert policy
    by_expert = defaultdict(list)
    for r in results:
        by_expert[r['expert_policy']].append(r)
    
    for expert in sorted(by_expert.keys()):
        print(f"\n{expert.upper()} Expert Policy:")
        print("-" * 120)
        print(f"{'Architecture':<15} {'Accuracy':<15} {'F1':<15} {'Precision':<15} {'Recall':<15} {'Slippage (bps)':<20}")
        print("-" * 120)
        
        for r in sorted(by_expert[expert], key=lambda x: x['architecture']):
            acc = f"{r['accuracy']:.5f}" if r['accuracy'] is not None else "N/A"
            f1 = f"{r['f1']:.5f}" if r['f1'] is not None else "N/A"
            prec = f"{r['precision']:.5f}" if r['precision'] is not None else "N/A"
            recall = f"{r['recall']:.5f}" if r['recall'] is not None else "N/A"
            slip = f"{r['slippage_bps']:.6f}" if r['slippage_bps'] is not None else "N/A"
            
            print(f"{r['architecture']:<15} {acc:<15} {f1:<15} {prec:<15} {recall:<15} {slip:<20}")
    
    # Summary stats
    print("\n" + "="*120)
    print("SUMMARY STATISTICS")
    print("="*120)
    
    for expert in sorted(by_expert.keys()):
        subset = by_expert[expert]
        print(f"\n{expert.upper()}:")
        print("-" * 60)
        
        # Calculate averages
        accuracies = [r['accuracy'] for r in subset if r['accuracy'] is not None]
        f1s = [r['f1'] for r in subset if r['f1'] is not None]
        slippages = [r['slippage_bps'] for r in subset if r['slippage_bps'] is not None]
        
        if accuracies:
            avg_acc = sum(accuracies) / len(accuracies)
            print(f"  Avg Accuracy: {avg_acc:.5f} (range: {min(accuracies):.5f} - {max(accuracies):.5f})")
        
        if f1s:
            avg_f1 = sum(f1s) / len(f1s)
            print(f"  Avg F1:       {avg_f1:.5f} (range: {min(f1s):.5f} - {max(f1s):.5f})")
        
        if slippages:
            avg_slip = sum(slippages) / len(slippages)
            print(f"  Avg Slippage: {avg_slip:.6f} bps (range: {min(slippages):.6f} - {max(slippages):.6f})")
        
        # Best architecture
        best_by_acc = max(subset, key=lambda r: r['accuracy'] or 0)
        if best_by_acc['accuracy'] is not None:
            print(f"  Best Accuracy: {best_by_acc['architecture']} ({best_by_acc['accuracy']:.5f})")
    
    print("\n" + "="*120)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare architecture runs from local logs')
    parser.add_argument('--results-dir', default='results/architecture_sweep', 
                        help='Results directory path')
    parser.add_argument('--csv', default=None, help='Save results to CSV file')
    
    args = parser.parse_args()
    
    # Extract metrics
    results = generate_comparison(args.results_dir)
    
    # Print table
    print_results(results)
    
    # Save to CSV if requested
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        
        fieldnames = ['expert_policy', 'architecture', 'run_name', 
                      'accuracy', 'f1', 'precision', 'recall', 'slippage_bps']
        
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n✓ Results saved to: {args.csv}")

if __name__ == '__main__':
    main()
