# ============================================================
# MODULE 5: Neural Network Diagnostic Model
# Covers: Week 10 (Neural Networks & Deep Learning)
# Trained on data/patient_records.csv (real project dataset)
#
# Uses scikit-learn's MLPClassifier rather than TensorFlow/Keras.
# TensorFlow isn't installable on every Python version (notably very
# new Python releases before TF publishes wheels for them) — MLPClassifier
# gives the same conceptual architecture (a multi-layer, fully-connected
# feed-forward network trained with Adam + early stopping) using only
# scikit-learn, which you already depend on for Module 4. The one thing
# sklearn's MLP genuinely can't do that Keras can is BatchNormalization
# and Dropout layers — there's no direct equivalent in MLPClassifier, so
# this is a real (and worth mentioning in your report) simplification,
# not just a renaming. L2 regularization (alpha) and early stopping are
# both still present and doing the same job they did in the Keras version.
# ============================================================

import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import List, Dict


def find_data_file(filename: str) -> str:
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


class NeuralDiagnosticModel:
    """
    Neural network diagnostic model (scikit-learn MLPClassifier).
    Architecture: Input -> 128 -> 64 -> 32 -> Output, ReLU, Adam,
    L2 regularization, early stopping on a held-out validation split.

    Trains directly on data/patient_records.csv. SYMPTOM_FEATURES and
    DISEASE_LABELS are populated from the CSV's actual columns/diagnoses
    the first time train() or predict() runs.
    """

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
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self._X_test = None
        self._y_test = None

    def _load_data(self):
        """Load data/patient_records.csv, set SYMPTOM_FEATURES/DISEASE_LABELS
        from its real columns, and return (X, y) as a DataFrame + Series
        (kept as raw strings for y — label-encoding happens in train())."""
        path = find_data_file("patient_records.csv")
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]

        symptom_cols = [c for c in df.columns if c not in ("Diagnosis", "Severity")]
        self.SYMPTOM_FEATURES = [c.lower() for c in symptom_cols]

        # NOTE: .to_numpy() instead of .values — see ml_classifier.py's
        # _load_data() for why .values is unsafe here (PyArrow-backed
        # pandas arrays can't be fancy-indexed by sklearn).
        X = df[symptom_cols].to_numpy(dtype=np.float32)
        y = df["Diagnosis"].to_numpy(dtype=object)
        return X, y

    def _build_model(self):
        """Build the MLPClassifier. Called once we know real class counts,
        so it's built at the start of train() rather than in __init__."""
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,               # L2 regularization
            batch_size=32,
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,       # holds out validation_fraction internally
            validation_fraction=0.2,
            n_iter_no_change=10,       # ~ patience=10, like the Keras version
            random_state=42,
        )

    def train(self, epochs: int = None, verbose: int = 1,
              X_train=None, y_train=None, X_test=None, y_test=None) -> Dict:
        """Train the neural network.

        Normally loads data/patient_records.csv and makes its own 80/20
        split. Pass X_train/y_train/X_test/y_test explicitly (as done by
        evaluation/metrics.py) to instead train on a specific split that's
        shared with the other modules, so cross-module comparisons are
        evaluated on the exact same held-out patients.

        `epochs` is accepted for interface-compatibility with the old
        Keras version; MLPClassifier's equivalent is max_iter, already
        set generously in _build_model().
        """
        self._build_model()

        if X_train is None:
            X, y = self._load_data()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)

        # Fit the label encoder on the union of train+test labels so
        # every class is known even if a class happens to be under-
        # represented in one split.
        import numpy as _np
        all_labels = _np.unique(_np.concatenate([y_train, y_test]))
        self.label_encoder.fit(all_labels)
        self.DISEASE_LABELS = list(self.label_encoder.classes_)

        y_train_enc = self.label_encoder.transform(y_train)
        y_test_enc = self.label_encoder.transform(y_test)

        if verbose:
            print("=" * 55)
            print("  Neural Network (MLPClassifier) — Training")
            print(f"  Architecture: {X_train.shape[1]} -> "
                  f"128 -> 64 -> 32 -> {len(self.DISEASE_LABELS)}")
            print("=" * 55)

        self.model.fit(X_train, y_train_enc)

        self._X_test, self._y_test = X_test, y_test_enc
        train_acc = self.model.score(X_train, y_train_enc)
        test_acc = self.model.score(X_test, y_test_enc)
        self.is_trained = True

        if verbose:
            print(f"  Train Accuracy: {train_acc:.4f}")
            print(f"  Test  Accuracy: {test_acc:.4f}")
            print(f"  Stopped after {self.model.n_iter_} iterations "
                  f"(early stopping, best_validation_score_="
                  f"{getattr(self.model, 'best_validation_score_', float('nan')):.4f})")

        return {'train_accuracy': train_acc, 'test_accuracy': test_acc}

    def predict(self, symptoms: List[str]) -> Dict:
        """Predict from symptom list"""
        if not self.is_trained:
            self.train(verbose=0)

        clean_symptoms = [s.lower().strip().replace(' ', '_') for s in symptoms]

        features = np.array([
            [1.0 if feat in clean_symptoms else 0.0
             for feat in self.SYMPTOM_FEATURES]
        ], dtype=np.float32)

        proba = self.model.predict_proba(features)[0]
        pred_idx = int(np.argmax(proba))
        diagnosis = self.DISEASE_LABELS[pred_idx]

        return {
            'diagnosis':  diagnosis,
            'confidence': round(float(proba[pred_idx]), 4),
            'all_probs':  dict(zip(self.DISEASE_LABELS,
                                   proba.round(4).tolist()))
        }

    def analyze(self, percept) -> Dict:
        """Module interface for the agent"""
        result = self.predict(percept.symptoms)
        result['summary'] = (f"NN: {result['diagnosis']} "
                             f"({result['confidence']:.2%})")
        return result

    def plot_training(self):
        """Plot training loss curve and validation accuracy over
        iterations (MLPClassifier's equivalent of Keras' history)."""
        if not self.is_trained:
            print("Train model first!")
            return

        has_val = hasattr(self.model, 'validation_scores_') and self.model.validation_scores_

        fig, axes = plt.subplots(1, 2 if has_val else 1, figsize=(14 if has_val else 7, 5))
        axes = axes if has_val else [axes]

        axes[0].plot(self.model.loss_curve_, color='#3498db', linewidth=2)
        axes[0].set_title("Training Loss", fontsize=13, fontweight='bold')
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)

        if has_val:
            axes[1].plot(self.model.validation_scores_, color='#2ecc71', linewidth=2)
            axes[1].set_title("Validation Accuracy", fontsize=13, fontweight='bold')
            axes[1].set_xlabel("Iteration")
            axes[1].set_ylabel("Accuracy")
            axes[1].grid(True, alpha=0.3)

        plt.suptitle("Neural Network (MLPClassifier) Training Curves",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("nn_training.png", dpi=150)
        plt.show()
# Module completed.