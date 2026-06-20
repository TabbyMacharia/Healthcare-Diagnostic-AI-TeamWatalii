# ============================================================
# MODULE 3: Bayesian Network — Probabilistic Diagnosis
# Covers: Week 7 (Bayesian Networks)
# ============================================================

import numpy as np
from typing import Dict, List

class SimpleBayesianDiagnostics:
    """
    Simplified Bayesian diagnostic model using
    pre-computed conditional probabilities.
    """

    def __init__(self):
        # Prior probabilities P(Disease)
        self.priors = {
            'flu':        0.15,
            'covid19':    0.08,
            'dengue':     0.05,
            'cardiac':    0.04,
            'diabetes':   0.10,
            'common_cold':0.30,
            'healthy':    0.28,
        }

        # Likelihoods: P(Symptom | Disease)
        # Format: disease -> {symptom -> P(symptom|disease)}
        self.likelihoods = {
            'flu': {
                'fever': 0.90, 'cough': 0.85, 'fatigue': 0.88,
                'headache': 0.70, 'body_aches': 0.80,
                'loss_of_smell': 0.20, 'chest_pain': 0.05,
                'rash': 0.05, 'joint_pain': 0.40,
            },
            'covid19': {
                'fever': 0.88, 'cough': 0.80, 'fatigue': 0.90,
                'loss_of_smell': 0.85, 'headache': 0.65,
                'body_aches': 0.60, 'chest_pain': 0.20,
                'rash': 0.05, 'joint_pain': 0.20,
            },
            'dengue': {
                'fever': 0.98, 'rash': 0.75, 'joint_pain': 0.85,
                'headache': 0.90, 'fatigue': 0.80,
                'cough': 0.15, 'loss_of_smell': 0.05,
                'chest_pain': 0.05, 'body_aches': 0.88,
            },
            'cardiac': {
                'chest_pain': 0.92, 'shortness_of_breath': 0.88,
                'fatigue': 0.70, 'sweating': 0.75,
                'fever': 0.10, 'cough': 0.15, 'rash': 0.02,
                'joint_pain': 0.10, 'headache': 0.30,
            },
            'diabetes': {
                'fatigue': 0.82, 'frequent_urination': 0.95,
                'excessive_thirst': 0.92, 'blurred_vision': 0.70,
                'fever': 0.10, 'cough': 0.05, 'rash': 0.08,
                'headache': 0.40, 'joint_pain': 0.20,
            },
            'common_cold': {
                'cough': 0.90, 'fever': 0.50, 'headache': 0.60,
                'fatigue': 0.55, 'body_aches': 0.50,
                'loss_of_smell': 0.30, 'rash': 0.02,
                'chest_pain': 0.05, 'joint_pain': 0.15,
            },
            'healthy': {
                'fever': 0.02, 'cough': 0.05, 'fatigue': 0.10,
                'headache': 0.08, 'rash': 0.01, 'chest_pain': 0.01,
                'joint_pain': 0.05, 'loss_of_smell': 0.01,
                'body_aches': 0.05,
            }
        }

    def compute_posterior(self,
                          symptoms: List[str]) -> Dict[str, float]:
        """
        Naïve Bayes posterior:
        P(D|S₁,...,Sₙ) ∝ P(D) × ∏ P(Sᵢ|D)
        """
        posteriors = {}
        symptoms_clean = [s.lower().replace(' ', '_') for s in symptoms]

        for disease, prior in self.priors.items():
            log_prob = np.log(prior)
            for symptom in symptoms_clean:
                p_s_given_d = self.likelihoods[disease].get(symptom, 0.01)
                log_prob += np.log(p_s_given_d)
            posteriors[disease] = log_prob

        # Convert log-probabilities to probabilities
        max_log = max(posteriors.values())
        exp_probs = {d: np.exp(v - max_log)
                     for d, v in posteriors.items()}
        total = sum(exp_probs.values())
        return {d: round(v/total, 4) for d, v in exp_probs.items()}

    def analyze(self, percept) -> Dict:
        """Module interface for the agent"""
        posteriors = self.compute_posterior(percept.symptoms)
        top_disease = max(posteriors, key=posteriors.get)
        top_prob    = posteriors[top_disease]
        sorted_dx   = sorted(posteriors.items(),
                             key=lambda x: x[1], reverse=True)

        return {
            'summary':    f"Top: {top_disease} ({top_prob:.2%})",
            'diagnosis':  top_disease,
            'confidence': top_prob,
            'all_posteriors': posteriors,
            'ranked_diagnoses': sorted_dx[:5]
        }

    def explain(self, disease: str, symptoms: List[str]) -> str:
        symptoms_clean = [s.lower().replace(' ','_') for s in symptoms]
        likelihoods    = self.likelihoods.get(disease, {})
        evidence = [
            f"P({s}|{disease})={likelihoods.get(s,0.01):.2f}"
            for s in symptoms_clean
        ]
        return f"P({disease}) = {self.priors[disease]} × " + \
               " × ".join(evidence)
