#!/usr/bin/env python3
"""
Simple script to view MLflow training data without starting the full UI
"""
import json
import os
from pathlib import Path

def read_mlflow_runs():
    mlruns_dir = Path('./mlruns_remote')
    
    if not mlruns_dir.exists():
        print("❌ No mlruns directory found!")
        return
    
    experiments = [d for d in mlruns_dir.iterdir() if d.is_dir() and d.name != '.trash']
    
    print("=" * 80)
    print("MLflow Training Runs Summary")
    print("=" * 80)
    
    for exp_dir in sorted(experiments):
        if exp_dir.name.startswith('.'):
            continue
            
        meta_file = exp_dir / 'meta.yaml'
        if not meta_file.exists():
            continue
            
        print(f"\n📊 Experiment: {exp_dir.name}")
        
        # Find all runs
        runs = [r for r in exp_dir.iterdir() if r.is_dir() and not r.name.startswith('.')]
        
        for run_dir in sorted(runs, key=lambda x: x.stat().st_mtime, reverse=True):
            meta = run_dir / 'meta.yaml'
            if not meta.exists():
                continue
                
            print(f"\n  🏃 Run: {run_dir.name}")
            print(f"     Modified: {run_dir.stat().st_mtime}")
            
            # Read metrics
            metrics_dir = run_dir / 'metrics'
            if metrics_dir.exists():
                print("\n     📈 Metrics:")
                for metric_file in sorted(metrics_dir.iterdir()):
                    with open(metric_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            # Show first and last value
                            first = lines[0].strip().split()
                            last = lines[-1].strip().split()
                            metric_name = metric_file.name
                            print(f"        {metric_name}:")
                            print(f"          First: {first[1]} (step {first[0]})")
                            print(f"          Last:  {last[1]} (step {last[0]})")
            
            # Read params
            params_dir = run_dir / 'params'
            if params_dir.exists():
                print("\n     ⚙️  Parameters:")
                for param_file in sorted(params_dir.iterdir())[:10]:  # First 10
                    with open(param_file, 'r') as f:
                        value = f.read().strip()
                        print(f"        {param_file.name}: {value}")
            
            print("\n" + "-" * 80)

if __name__ == '__main__':
    read_mlflow_runs()
