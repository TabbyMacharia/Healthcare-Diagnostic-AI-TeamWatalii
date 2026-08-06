# ============================================================
# Evaluation Module — Visualizations
# Reusable plotting functions used by evaluation/metrics.py
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, labels, title, save_path):
    """Save a single confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={'label': 'Count'})
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_all_confusion_matrices(cm_dict, labels, save_path):
    """cm_dict: {module_name: confusion_matrix}. One figure, one subplot
    per module, side by side — used for the report's "confusion matrices
    for each classifier" deliverable in a single glance-able image."""
    n = len(cm_dict)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (name, cm) in zip(axes, cm_dict.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    cbar=False)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.suptitle("Confusion Matrices — All Diagnostic Modules",
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_module_comparison(metrics_by_module, save_path):
    """metrics_by_module: {module_name: {'accuracy':.., 'precision':..,
    'recall':.., 'f1':..}}. Grouped bar chart comparing all 4 modules
    across all 4 metrics — the "Module comparison bar chart" deliverable."""
    modules = list(metrics_by_module.keys())
    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(modules))
    width = 0.2
    colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, metric in enumerate(metric_names):
        values = [metrics_by_module[m][metric] for m in modules]
        ax.bar(x + i * width, values, width, label=metric.capitalize(),
               color=colors[i])

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(modules)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Module Comparison — Accuracy / Precision / Recall / F1",
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_roc_curves(roc_data, save_path):
    """roc_data: {module_name: (fpr_macro, tpr_macro, auc_macro)} —
    one macro-average ROC curve per module, overlaid on one axes."""
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']

    for (name, (fpr, tpr, auc)), color in zip(roc_data.items(), colors):
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (macro-average, one-vs-rest) — All Modules",
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")