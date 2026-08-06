# ============================================================
# MODULE 3: Bayesian Network — Probabilistic Diagnosis
# Covers: Week 7 (Bayesian Networks & Naïve Bayes)
# Priors and likelihoods are LEARNED from data/patient_records.csv
# rather than hand-guessed, so they reflect the real dataset.
# ============================================================

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any


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


class SimpleBayesianDiagnostics:
    """
    Bayesian diagnostic model using conditional probabilities learned
    directly from data/patient_records.csv, combined with Naive Bayes
    logic in log space.

    self.priors[disease]              = P(disease)  — from class frequency
    self.likelihoods[disease][symptom] = P(symptom | disease) — from
                                          the fraction of that disease's
                                          patients who have the symptom
                                          (Laplace-smoothed so nothing is
                                          ever exactly 0 or 1)
    """

    def __init__(self, data_path: str = None, train_df: "pd.DataFrame" = None):
        """
        train_df: optionally pass a pre-loaded, pre-split DataFrame (e.g.
        only the training partition) to learn priors/likelihoods from —
        used by evaluation/metrics.py so this module doesn't "peek" at
        held-out test patients when computing its probabilities. If not
        given, falls back to learning from the whole CSV (the standalone,
        per-module-testing use case the lab manual describes).
        """
        self.data_path = data_path or find_data_file("patient_records.csv")
        self.priors: Dict[str, float] = {}
        self.likelihoods: Dict[str, Dict[str, float]] = {}
        self.symptom_features: List[str] = []
        self._learn_from_data(train_df)

    def _learn_from_data(self, train_df: "pd.DataFrame" = None):
        df = train_df if train_df is not None else pd.read_csv(self.data_path)
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]

        symptom_cols = [c for c in df.columns if c not in ("Diagnosis", "Severity")]
        self.symptom_features = [c.lower() for c in symptom_cols]

        counts = df["Diagnosis"].value_counts()
        total = counts.sum()
        self.priors = {disease: count / total for disease, count in counts.items()}

        self.likelihoods = {}
        for disease in counts.index:
            subset = df[df["Diagnosis"] == disease]
            n = len(subset)
            self.likelihoods[disease] = {}
            for col, feat in zip(symptom_cols, self.symptom_features):
                # Laplace (add-one) smoothing: avoids exact 0.0/1.0 so
                # log-space math never hits log(0), and avoids being
                # overconfident from a small subset.
                p = (subset[col].sum() + 1) / (n + 2)
                self.likelihoods[disease][feat] = round(float(p), 4)

    def compute_posterior(self, symptoms: List[str]) -> Dict[str, float]:
        """
        Computes Naive Bayes Posterior Probability:
        P(D | S1, ..., Sn) proportional to P(D) * prod P(Si | D)
        Uses log-space additions to prevent underflow error.
        """
        posteriors = {}
        symptoms_clean = [s.lower().strip().replace(' ', '_') for s in symptoms]

        for disease, prior in self.priors.items():
            log_prob = np.log(prior)
            for symptom in symptoms_clean:
                # 0.01 floor for symptoms outside our known feature set
                p_s_given_d = self.likelihoods[disease].get(symptom, 0.01)
                log_prob += np.log(p_s_given_d)
            posteriors[disease] = log_prob

        # Softmax-style conversion back to normalized probabilities
        max_log = max(posteriors.values())
        exp_probs = {d: np.exp(v - max_log) for d, v in posteriors.items()}
        total = sum(exp_probs.values())

        return {d: round(v / total, 4) for d, v in exp_probs.items()}

    def analyze(self, percept: Any) -> Dict[str, Any]:
        """Standard agent interface method."""
        symptoms = getattr(percept, 'symptoms', [])
        posteriors = self.compute_posterior(symptoms)

        sorted_dx = sorted(posteriors.items(), key=lambda x: x[1], reverse=True)
        top_disease, top_prob = sorted_dx[0]

        return {
            'summary': f"Top: {top_disease} ({top_prob:.2%})",
            'diagnosis': top_disease,
            'confidence': top_prob,
            'all_posteriors': posteriors,
            'ranked_diagnoses': sorted_dx[:5]
        }

    def explain(self, disease: str, symptoms: List[str]) -> str:
        """Human-readable breakdown of the posterior logic for a disease."""
        symptoms_clean = [s.lower().strip().replace(' ', '_') for s in symptoms]
        likelihoods = self.likelihoods.get(disease, {})

        evidence = [
            f"P({s}|{disease})={likelihoods.get(s, 0.01):.2f}"
            for s in symptoms_clean
        ]

        prior_val = self.priors.get(disease, 0.0)
        return f"P({disease}) = {prior_val:.4f} x " + " x ".join(evidence)


# ============================================================
# MODULE TESTER / DEMO
# ============================================================
if __name__ == "__main__":
    bn = SimpleBayesianDiagnostics()

    print(f"Learned priors for {len(bn.priors)} diseases from data/patient_records.csv:")
    for disease, p in sorted(bn.priors.items(), key=lambda x: -x[1]):
        print(f"  {disease:<15}: {p:.2%}")

    test_symptoms = ["fever", "cough", "headache", "fatigue", "chills", "body_ache"]
    print(f"\nTesting symptoms: {test_symptoms}\n")

    posteriors = bn.compute_posterior(test_symptoms)
    print("Ranked Diagnoses:")
    ranked = sorted(posteriors.items(), key=lambda x: x[1], reverse=True)
    for disease, prob in ranked:
        print(f"  - {disease:<15}: {prob:.2%}")

    print("\nExplanation breakdown for top disease:")
    top_disease = ranked[0][0]
    print(" ", bn.explain(top_disease, test_symptoms))