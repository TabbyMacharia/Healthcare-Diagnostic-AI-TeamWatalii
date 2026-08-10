# ============================================================
# MODULE 4: ML Classifier — Supervised Diagnosis
# Covers: Week 9 (Supervised Learning & Decision Trees)
# Trained on data/patient_records.csv (real project dataset)
# ============================================================

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def find_data_file(filename: str) -> str:
    """Locate a data/ CSV whether this module is run from the project
    root, from inside modules/, or imported as a package. Tries the
    project layout from the lab manual (data/ as a sibling of modules/)
    first, then falls back to a few common alternatives."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "data", filename),
        os.path.join(here, "data", filename),
        os.path.join("data", filename),
        filename,
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"Could not find '{filename}'. Expected it inside a data/ folder "
        f"next to modules/ (see the lab manual's project structure)."
    )


class MLDiagnosticClassifier:
    """
    Ensemble ML-based diagnostic classifier.
    Uses Decision Trees, Random Forest, and
    Gradient Boosting for robust diagnosis.

    Trains directly on data/patient_records.csv rather than synthetic
    data — SYMPTOM_FEATURES and DISEASE_LABELS are populated from the
    CSV's actual columns and Diagnosis values the first time train()
    or predict() is called.
    """

    # Populated dynamically from the CSV in _load_data(); kept as class
    # defaults so other code (e.g. predict() before training) has sane
    # fallbacks that match data/patient_records.csv's real columns.
    SYMPTOM_FEATURES = [
        'fever', 'cough', 'headache', 'fatigue', 'sore_throat',
        'chest_pain', 'shortness_of_breath', 'nausea', 'vomiting',
        'diarrhea', 'body_ache', 'runny_nose', 'sneezing',
        'loss_of_taste', 'loss_of_smell', 'chills', 'dizziness',
        'high_blood_pressure', 'low_blood_pressure', 'high_heart_rate'
    ]

    DISEASE_LABELS = [
        'COVID-19', 'Common Cold', 'Food Poisoning', 'Healthy',
        'Hypertension', 'Influenza', 'Malaria', 'Migraine',
        'Pneumonia', 'Typhoid'
    ]

    def __init__(self):
        self.models = {
            'Decision Tree':     DecisionTreeClassifier(
                max_depth=8, criterion='entropy', random_state=42),
            'Random Forest':     RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, random_state=42),
        }
        self.best_model    = None
        self.best_model_name = None
        self.label_encoder = LabelEncoder()
        self.is_trained    = False
        self._features_set = False

    def _load_data(self) -> pd.DataFrame:
        """Load the real project dataset from data/patient_records.csv.

        The CSV's symptom columns are Title_Case (e.g. 'Sore_Throat');
        SYMPTOM_FEATURES/predict() use lower_snake_case, matching the
        symptom-cleaning convention used everywhere else in this project
        ("Loss of Smell" -> "loss_of_smell"). We lowercase the columns
        here so both sides agree.
        """
        path = find_data_file("patient_records.csv")
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]

        symptom_cols = [c for c in df.columns if c not in ("Diagnosis", "Severity")]
        self.SYMPTOM_FEATURES = [c.lower() for c in symptom_cols]
        self.DISEASE_LABELS = sorted(df["Diagnosis"].unique().tolist())
        self._features_set = True

        # Rename to lower_snake_case so downstream code (predict(), etc.)
        # can index columns by the same names used in SYMPTOM_FEATURES.
        df = df.rename(columns={c: c.lower() for c in symptom_cols})
        return df

    def train(self, verbose: bool = True,
              X_train=None, X_test=None, y_train=None, y_test=None) -> Dict:
        """Train all models and select the best one.

        By default, loads data/patient_records.csv and makes its own
        80/20 split (the standalone, per-module-testing use case the lab
        manual describes). Optionally, pass a pre-made X_train/X_test/
        y_train/y_test (raw symptom arrays + raw Diagnosis strings) —
        used by evaluation/metrics.py so ML classifier, Neural Network,
        Bayesian Net and Knowledge Base can all be scored on the exact
        same held-out patients for a fair module comparison.
        """
        if X_train is None:
            df = self._load_data()
            X = df[self.SYMPTOM_FEATURES].to_numpy(dtype=float)
            # NOTE: bug fixed here — .values (used to work fine) can
            # silently return a PyArrow-backed array instead of a plain
            # numpy array when the `pyarrow` package is installed
            # alongside certain pandas versions. sklearn's train_test_split
            # can't fancy-index that array type and fails with
            # "only integer scalar arrays can be converted to a scalar
            # index" deep inside pyarrow. .to_numpy() forces a real numpy
            # array regardless of pandas' internal backend, which .values
            # does not guarantee.
            y_raw = df["Diagnosis"].to_numpy(dtype=object)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_raw, test_size=0.2, random_state=42, stratify=y_raw)
        else:
            # SYMPTOM_FEATURES/DISEASE_LABELS still need to be correct
            # even when we skip _load_data() — pull them once so
            # predict() and plot_evaluation() work normally afterwards.
            if not hasattr(self, "_features_set") or not self._features_set:
                self._load_data()

        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_test_enc  = self.label_encoder.transform(y_test)

        results = {}
        best_acc = 0.0

        if verbose:
            print("=" * 55)
            print("  ML Diagnostic Classifier — Training")
            print("=" * 55)

        for name, model in self.models.items():
            model.fit(X_train, y_train_enc)
            # Cross-validate on the TRAIN fold only — using the full
            # X/y here (as the original code did) would leak the held-out
            # test patients into cross-validation.
            cv_scores = cross_val_score(model, X_train, y_train_enc, cv=5, scoring='accuracy')
            test_acc  = model.score(X_test, y_test_enc)
            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std':  cv_scores.std(),
                'test_acc': test_acc
            }
            if verbose:
                print(f"\n  Model: {name}")
                print(f"     CV Accuracy : {cv_scores.mean():.4f} "
                      f"± {cv_scores.std():.4f}")
                print(f"     Test Accuracy: {test_acc:.4f}")

            if test_acc > best_acc:
                best_acc          = test_acc
                self.best_model   = model
                self.best_model_name = name

        self.is_trained = True
        self._X_test = X_test
        self._y_test = y_test_enc

        if verbose:
            print(f"\n  Best Model: {self.best_model_name} "
                  f"({best_acc:.4f})")
        return results

    def predict(self, symptoms: List[str]) -> Dict:
        """Predict disease from symptom list"""
        if not self.is_trained:
            self.train(verbose=False)

        symptoms = [s.lower().strip() for s in symptoms
                    if s.lower().strip() in self.SYMPTOM_FEATURES]

        features = np.array([
            [1 if s in symptoms else 0
             for s in self.SYMPTOM_FEATURES]
        ])
        pred_encoded = self.best_model.predict(features)[0]
        pred_proba   = self.best_model.predict_proba(features)[0]

        disease  = self.label_encoder.inverse_transform([pred_encoded])[0]
        classes  = self.label_encoder.inverse_transform(
            range(len(pred_proba)))
        prob_map = dict(zip(classes, pred_proba))
        top5     = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            'diagnosis':      disease,
            'confidence':     round(float(pred_proba[pred_encoded]), 4),
            'top5':           top5,
            'all_probs':      {k: round(float(v), 4) for k, v in prob_map.items()},
            'model_used':     self.best_model_name,
            'symptom_vector': features[0].tolist()
        }

    def analyze(self, percept) -> Dict:
        """Module interface for the agent"""
        result = self.predict(percept.symptoms)
        result['summary'] = (f"{result['model_used']}: "
                             f"{result['diagnosis']} "
                             f"({result['confidence']:.2%})")
        return result

    def plot_evaluation(self):
        """Visualize model performance"""
        if not self.is_trained:
            self.train(verbose=False)

        y_pred = self.best_model.predict(self._X_test)
        print(classification_report(
            self._y_test,
            y_pred,
            target_names=self.label_encoder.classes_))

        cm     = confusion_matrix(self._y_test, y_pred)
        labels = self.label_encoder.classes_

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=axes[0])
        axes[0].set_title(f"Confusion Matrix\n({self.best_model_name})",
                          fontweight='bold')
        axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Feature Importance
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            sorted_idx  = np.argsort(importances)[::-1][:12]
            top_features = [self.SYMPTOM_FEATURES[i] for i in sorted_idx]
            top_values   = importances[sorted_idx]
            colors = plt.cm.RdYlGn(top_values / top_values.max())
            axes[1].barh(range(len(top_features)), top_values[::-1],
                         color=colors[::-1])
            axes[1].set_yticks(range(len(top_features)))
            axes[1].set_yticklabels(top_features[::-1])
            axes[1].set_title("Feature Importances (Top 12)",
                              fontweight='bold')
            axes[1].set_xlabel("Importance Score")

        plt.suptitle(f"ML Diagnostic Model Evaluation — {self.best_model_name}",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("ml_evaluation.png", dpi=150, bbox_inches="tight")

        # Display the graph once
        plt.show(block=True)

        # Close all figure windows after the user closes the graph
        plt.close("all")

        print("Saved: ml_evaluation.png")