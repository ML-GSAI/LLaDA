"""
Analyze and visualize results from run_trajectory_uq.py
Usage: python3 analyze_results.py results/triviaqa_traj_uq.json
"""

import json
import sys
import numpy as np

def main(result_file):
    with open(result_file) as f:
        data = json.load(f)

    print("=" * 60)
    print("Trajectory-Based UQ for LLaDA — TriviaQA Results")
    print("=" * 60)
    print(f"Model:      {data['args']['model_name']}")
    print(f"N examples: {data['n_total']}")
    print(f"Steps:      {data['args']['steps']}")
    print(f"Gen length: {data['args']['gen_length']}")
    print(f"Accuracy:   {data['accuracy']:.3f} ({data['n_correct']}/{data['n_total']})")
    print()

    print("Uncertainty Metric Evaluation:")
    print(f"  {'Metric':<22} {'AUROC':>8} {'ECE':>8}")
    print("  " + "-" * 40)
    for metric, vals in data['eval_results'].items():
        print(f"  {metric:<22} {vals['auroc']:>8.4f} {vals['ece']:>8.4f}")

    print()
    print("Sample predictions (first 5):")
    for s in data['samples'][:5]:
        status = "✓" if s['correct'] else "✗"
        print(f"  [{status}] Q: {s['question'][:60]}")
        print(f"      Gen: {s['generated'][:60]}")
        print(f"      mean_entropy={s['mean_entropy']:.3f}  traj_var={s['traj_variance']:.3f}")
        print()

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results/triviaqa_traj_uq.json')
