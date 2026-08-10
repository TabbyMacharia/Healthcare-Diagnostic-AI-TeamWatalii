# ============================================================
# Evaluation Module — Metrics
# Covers the lab manual's "Evaluation Module" deliverables:
# Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC —
# computed for all 4 diagnostic modules (Knowledge Base, Bayesian
# Net, ML Classifier, Neural Network) on ONE shared, held-out test
# set, so the module-comparison numbers are a fair apples-to-apples
# comparison rather than each module grading its own homework on
# whatever split it happened to make internally.
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from modules.knowledge_base import MedicalKnowledgeBase
from modules.bayesian_net import SimpleBayesianDiagnostics
from modules.ml_classifier import MLDiagnosticClassifier
from modules.neural_network import NeuralDiagnosticModel

from evaluation.visualizations import (
    plot_confusion_matrix, plot_all_confusion_matrices,
    plot_module_comparison, plot_roc_curves
)


def find_data_file(filename: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "data", filename),
        os.path.join("data", filename),
        filename,
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Could not find '{filename}'.")


@dataclass
class EvalPercept:
    """Minimal stand-in for PatientPercept — just enough for each
    module's analyze() method (symptoms/temperature/heart_rate)."""
    symptoms: List[str]
    temperature: float
    heart_rate: int


def load_canonical_split(test_size=0.2, random_state=42):
    """The ONE train/test split every module is evaluated against."""
    path = find_data_file("patient_records.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    symptom_cols = [c for c in df.columns if c not in ("Diagnosis", "Severity")]
    y = df["Diagnosis"].to_numpy(dtype=object)

    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=test_size,
        random_state=random_state, stratify=y)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    X_train = train_df[symptom_cols].to_numpy(dtype=float)
    X_test = test_df[symptom_cols].to_numpy(dtype=float)
    y_train = train_df["Diagnosis"].to_numpy(dtype=object)
    y_test = test_df["Diagnosis"].to_numpy(dtype=object)

    return train_df, test_df, symptom_cols, X_train, X_test, y_train, y_test


def rows_to_percepts(test_df, symptom_cols):
    """Turn each test-set row into an EvalPercept (symptom list + rough
    vitals) so KB/Bayes/ML/NN can all be queried through their normal
    analyze() interface, exactly as the Agent would call them."""
    percepts = []
    for _, row in test_df.iterrows():
        symptoms = [c.lower() for c in symptom_cols if row[c] == 1]
        temperature = 38.6 if row.get("Fever", 0) == 1 else 36.8
        heart_rate = 105 if row.get("High_Heart_Rate", 0) == 1 else 78
        percepts.append(EvalPercept(symptoms=symptoms, temperature=temperature,
                                     heart_rate=heart_rate))
    return percepts


def evaluate_predictions(y_true, y_pred, class_labels):
    """accuracy/precision/recall/f1 (macro) + confusion matrix, all
    computed against a fixed canonical class ordering so confusion
    matrices line up across modules."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}, cm


def proba_dict_to_matrix(proba_dicts, class_labels):
    """Convert a list of {class_name: probability} dicts into an
    (n_samples, n_classes) matrix in canonical class order, for
    roc_auc_score(multi_class='ovr')."""
    matrix = np.zeros((len(proba_dicts), len(class_labels)))
    for i, d in enumerate(proba_dicts):
        for j, label in enumerate(class_labels):
            matrix[i, j] = d.get(label, 0.0)
        row_sum = matrix[i].sum()
        if row_sum > 0:
            matrix[i] /= row_sum  # renormalize in case of missing classes
    return matrix


def kb_proba_dict(inferred, top_diagnosis, confidence, class_labels):
    """KB is rule-based, not probabilistic, so it doesn't naturally
    produce a distribution over all 10 diseases the way Bayes/ML/NN do.
    For ROC-AUC purposes only, we approximate one: the predicted class
    gets its rule confidence, everything else splits the remainder
    evenly. This is a real approximation — call it out as a caveat
    when reporting KB's ROC-AUC in the report, it is not on the same
    footing as the other three modules' genuine probability outputs.
    """
    d = {label: 0.0 for label in class_labels}
    d[top_diagnosis] = confidence
    remainder = (1 - confidence) / max(len(class_labels) - 1, 1)
    for label in class_labels:
        if label != top_diagnosis:
            d[label] = remainder
    return d


def run_evaluation(output_dir=None):
    output_dir = output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "reports")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("  EVALUATION MODULE — Building canonical train/test split")
    print("=" * 65)
    train_df, test_df, symptom_cols, X_train, X_test, y_train, y_test = \
        load_canonical_split()
    print(f"  Train: {len(train_df)} patients | Test: {len(test_df)} patients")

    class_labels = sorted(pd.concat([train_df['Diagnosis'], test_df['Diagnosis']]).unique())
    print(f"  Classes ({len(class_labels)}): {class_labels}\n")

    percepts = rows_to_percepts(test_df, symptom_cols)

    # ---- Knowledge Base (rule-based, no training) ----
    print("Evaluating Knowledge Base...")
    kb = MedicalKnowledgeBase()
    kb_preds, kb_probas = [], []
    for p in percepts:
        r = kb.analyze(p)
        kb_preds.append(r['diagnosis'])
        kb_probas.append(kb_proba_dict(r['all_inferred'], r['diagnosis'],
                                        r['confidence'], class_labels))

    # ---- Bayesian Network (fit on TRAIN partition only — no leakage) ----
    print("Evaluating Bayesian Network...")
    bn = SimpleBayesianDiagnostics(train_df=train_df)
    bn_preds, bn_probas = [], []
    for p in percepts:
        r = bn.analyze(p)
        bn_preds.append(r['diagnosis'])
        bn_probas.append(r['all_posteriors'])

    # ---- ML Classifier (trained on the SAME canonical split) ----
    print("Evaluating ML Classifier...")
    ml = MLDiagnosticClassifier()
    ml.train(verbose=False, X_train=X_train, X_test=X_test,
             y_train=y_train, y_test=y_test)
    ml_preds, ml_probas = [], []
    for p in percepts:
        r = ml.predict(p.symptoms)
        ml_preds.append(r['diagnosis'])
        ml_probas.append(r['all_probs'])

    # ---- Neural Network (trained on the SAME canonical split) ----
    print("Evaluating Neural Network...")
    nn = NeuralDiagnosticModel()
    nn.train(verbose=0, X_train=X_train, y_train=y_train,
              X_test=X_test, y_test=y_test)
    nn_preds, nn_probas = [], []
    for p in percepts:
        r = nn.predict(p.symptoms)
        nn_preds.append(r['diagnosis'])
        nn_probas.append(r['all_probs'])

    modules = {
        'KnowledgeBase': (kb_preds, kb_probas),
        'BayesianNet':   (bn_preds, bn_probas),
        'MLClassifier':  (ml_preds, ml_probas),
        'NeuralNetwork': (nn_preds, nn_probas),
    }

    metrics_by_module = {}
    cm_by_module = {}
    roc_by_module = {}

    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)
    header = f"{'Module':<15}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}{'ROC-AUC':>10}"
    print(header)
    print("-" * len(header))

    for name, (preds, probas) in modules.items():
        metrics, cm = evaluate_predictions(y_test, preds, class_labels)
        metrics_by_module[name] = metrics
        cm_by_module[name] = cm

        proba_matrix = proba_dict_to_matrix(probas, class_labels)
        y_test_bin = label_binarize(y_test, classes=class_labels)
        try:
            auc = roc_auc_score(y_test_bin, proba_matrix, average='macro', multi_class='ovr')
        except ValueError:
            auc = float('nan')  # can happen if a class has 0 test examples
        metrics['roc_auc'] = auc

        fpr_points, tpr_points = _macro_roc_points(y_test_bin, proba_matrix)
        roc_by_module[name] = (fpr_points, tpr_points, auc)

        print(f"{name:<15}{metrics['accuracy']:>10.4f}{metrics['precision']:>11.4f}"
              f"{metrics['recall']:>9.4f}{metrics['f1']:>8.4f}{auc:>10.4f}")

    print()
    print("NOTE: KnowledgeBase's ROC-AUC is an approximation — it is a")
    print("rule-based module, not a probabilistic one, so its per-class")
    print("probabilities are synthesized from its single confidence score")
    print("(see kb_proba_dict() in this file) rather than genuinely learned.")

    # ---- Save visualizations ----
    print()
    for name, cm in cm_by_module.items():
        plot_confusion_matrix(cm, class_labels, f"Confusion Matrix — {name}",
                               os.path.join(output_dir, f"confusion_matrix_{name}.png"))
    plot_all_confusion_matrices(cm_by_module, class_labels,
                                 os.path.join(output_dir, "confusion_matrices_all.png"))
    plot_module_comparison(metrics_by_module,
                            os.path.join(output_dir, "module_comparison.png"))
    plot_roc_curves(roc_by_module, os.path.join(output_dir, "roc_curves.png"))

    # Also let ML classifier's own confusion-matrix/feature-importance
    # plot run, using the SAME canonical test set it was just trained on.
    ml.plot_evaluation()

    return metrics_by_module, cm_by_module, class_labels


def _macro_roc_points(y_test_bin, proba_matrix, n_points=100):
    """Macro-average ROC curve (interpolated to a common set of FPR
    points) across all classes, for a single overlay-able curve."""
    from sklearn.metrics import roc_curve
    n_classes = y_test_bin.shape[1]
    mean_fpr = np.linspace(0, 1, n_points)
    tprs = []
    for c in range(n_classes):
        if y_test_bin[:, c].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_test_bin[:, c], proba_matrix[:, c])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0) if tprs else np.zeros_like(mean_fpr)
    return mean_fpr, mean_tpr


if __name__ == "__main__":
    run_evaluation()