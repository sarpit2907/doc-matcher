import os
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def evaluate_all_runs():
    print("\n" + "="*60)
    print("STEP 1: Automatically Evaluating All Completed Runs")
    print("="*60)
    output_dir = Path('output')
    if not output_dir.exists():
        print("No output directory found. Run inference first.")
        return

    runs = sorted([d.name for d in output_dir.iterdir() if d.is_dir()])
    for run_name in runs:
        print(f"\nEvaluating: {run_name}")
        subprocess.run(["python", "eval.py", "--run", run_name], check=False)

def generate_qualitative():
    print("\n" + "="*60)
    print("STEP 2: Generating Qualitative Visual Report")
    print("="*60)
    
    models = {
        'Identity\n(No Dewarping)': 'output/inv3d_real-identity',
        'DewarpNet': 'output/inv3d_real-dewarpnet@inv3d',
        'GeoTr': 'output/inv3d_real-geotr@inv3d',
        'GeoTr\n+Template': 'output/inv3d_real-geotr_template@inv3d',
        'DocMatcher\n(Baseline)': 'output/inv3d_real-docmatcher@inv3d',
        'DocMatcher\n(GNN Proposed)': 'output/inv3d_real-docmatcher_gnn@inv3d', # Add hypothetical name
    }

    available_models = {k: v for k, v in models.items() if Path(v).exists()}
    
    if not available_models:
         # Fallback to whatever is in output if exact names fail
         output_dir = Path('output')
         available_models = {d.name: str(d) for d in output_dir.iterdir() if d.is_dir() and 'inv3d' in d.name}

    if not available_models:
        print("No models found to visualize.")
        return

    first_model_dir = Path(list(available_models.values())[0])
    sample_dirs = sorted([d for d in first_model_dir.iterdir() if d.is_dir()])

    if not sample_dirs:
        print("No actual image samples generated inside the model output directories yet.")
        return

    num_display = min(6, len(sample_dirs))
    random.seed(42)  # Keep images consistent for report
    selected_samples = random.sample(sample_dirs, num_display)

    num_models = len(available_models)
    fig, axes = plt.subplots(num_display, num_models, figsize=(4 * num_models, 4 * num_display))
    if num_display == 1:
        axes = axes.reshape(1, -1)

    for col, (model_name, model_dir) in enumerate(available_models.items()):
        axes[0, col].set_title(model_name, fontsize=14, fontweight='bold',
                               color='green' if 'Proposed' in model_name else 'black')

    for row, sample_path in enumerate(selected_samples):
        sample_name = sample_path.name
        for col, (model_name, model_dir) in enumerate(available_models.items()):
            ax = axes[row, col]
            sample_dir = Path(model_dir) / sample_name
            norm_files = list(sample_dir.glob('norm_image.*')) if sample_dir.exists() else []
            if norm_files:
                ax.imshow(mpimg.imread(str(norm_files[0])))
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    plt.suptitle('Document Dewarping — Model Comparison', fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('output/qualitative_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved qualitative grid to output/qualitative_comparison.png")

def generate_quantitative():
    print("\n" + "="*60)
    print("STEP 3: Generating Quantitative Charts & Tables")
    print("="*60)
    
    output_dir = Path('output')
    results = []

    model_display_names = {
        'inv3d_real-identity': 'Identity',
        'inv3d_real-dewarpnet@inv3d': 'DewarpNet',
        'inv3d_real-geotr@inv3d': 'GeoTr',
        'inv3d_real-geotr_template@inv3d': 'GeoTr + Template',
        'inv3d_real-docmatcher@inv3d': 'DocMatcher',
    }

    for run_dir in sorted(output_dir.iterdir()):
        if not run_dir.is_dir(): continue
        
        # Original DocMatcher format creates scores.csv or results.csv 
        # Attempt to read all possible outputs
        summary_file = run_dir / 'results_summary.csv'
        score_file = run_dir / 'score.csv'
        scores_file = run_dir / 'scores.csv'
        
        target_file = None
        if summary_file.exists(): target_file = summary_file
        elif score_file.exists(): target_file = score_file
        elif scores_file.exists(): target_file = scores_file

        if target_file:
            try:
                summary = pd.read_csv(target_file, header=None, index_col=0)
                row = {'Model': model_display_names.get(run_dir.name, run_dir.name)}
                for metric, value in summary.iterrows():
                    row[metric] = value.iloc[0]
                results.append(row)
            except Exception as e:
                print(f"Could not parse {target_file.name} for {run_dir.name}: {e}")

    if results:
        df = pd.DataFrame(results).set_index('Model')
        print(df.to_string())
        df.to_csv('output/comparison_results.csv')
        print("\nSaved raw tables to output/comparison_results.csv")
        
        metrics = df.columns.tolist()
        higher_is_better = {'ms_ssim', 'Precision', 'Recall', 'F1-Score'}
        colors = ['#95a5a6', '#e74c3c', '#3498db', '#9b59b6', '#2ecc71']

        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 6))
        if len(metrics) == 1: axes = [axes]

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            # Ensure values are numeric
            values = pd.to_numeric(df[metric], errors='coerce').fillna(0).values
            model_names = list(df.index)
            
            best_idx = np.argmax(values) if metric in higher_is_better else np.argmin(values)
            arrow = 'Higher' if metric in higher_is_better else 'Lower'
            bar_colors = [colors[i % len(colors)] for i in range(len(values))]
            bar_colors[best_idx] = '#27ae60'

            bars = ax.barh(model_names, values, color=bar_colors, edgecolor='white', linewidth=0.5)
            for bar, val in zip(bars, values):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{val:.4f}', va='center', fontsize=10)
            ax.set_title(f'{metric.upper()}\n({arrow} is better)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Score', fontsize=12)
            ax.invert_yaxis()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.suptitle('Quantitative Comparison — Inv3DReal Dataset', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('output/quantitative_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved Bar Charts to output/quantitative_comparison.png")
    else:
        print("No CSV evaluation results found. Make sure eval.py generated them.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-eval', action='store_true', help="Skip step 1 if eval.py already ran")
    args = parser.parse_args()
    
    if not args.skip_eval:
        evaluate_all_runs()
        
    generate_qualitative()
    generate_quantitative()
    
    print("\nALL REPORTS GENERATED! Look inside your output/ folder.")
