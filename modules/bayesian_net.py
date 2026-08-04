# ============================================================
# MODULE 3: Bayesian Network — Probabilistic Diagnosis
# Covers: Week 7 (Bayesian Networks & Naïve Bayes)
# ============================================================

import numpy as np
from typing import Dict, List, Any


class SimpleBayesianDiagnostics:
    """
    Simplified Bayesian diagnostic model using pre-computed conditional
    probabilities and Naïve Bayes logic in log space.
    """

    def __init__(self):
        # Prior probabilities P(Disease) - Base rate in population
        self.priors: Dict[str, float] = {
            'flu':         0.15,
            'covid19':     0.08,
            'dengue':      0.05,
            'cardiac':     0.04,
            'diabetes':    0.10,
            'common_cold': 0.30,
            'healthy':     0.28,
        }

        # Likelihoods: P(Symptom | Disease)
        # Format: disease -> {symptom -> P(symptom|disease)}
        self.likelihoods: Dict[str, Dict[str, float]] = {
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

    def compute_posterior(self, symptoms: List[str]) -> Dict[str, float]:
        """
        Computes Naïve Bayes Posterior Probability:
        P(D | S1, ..., Sn) ∝ P(D) * ∏ P(Si | D)
        Uses log-space additions to prevent underflow error.
        """
        posteriors = {}
        # Clean string inputs to match lower_snake_case formatting
        symptoms_clean = [s.lower().strip().replace(' ', '_') for s in symptoms]

        for disease, prior in self.priors.items():
            # Start with log of the prior P(Disease)
            log_prob = np.log(prior)
            
            for symptom in symptoms_clean:
                # Use floor value 0.01 for unseen symptoms to avoid log(0) undefined errors
                p_s_given_d = self.likelihoods[disease].get(symptom, 0.01)
                log_prob += np.log(p_s_given_d)
                
            posteriors[disease] = log_prob

        # Softmax conversion: convert log-probabilities back to standard probabilities
        max_log = max(posteriors.values())
        exp_probs = {d: np.exp(v - max_log) for d, v in posteriors.items()}
        total = sum(exp_probs.values())

        # Normalize so all disease probabilities sum to 1.0
        return {d: round(v / total, 4) for d, v in exp_probs.items()}

    def analyze(self, percept: Any) -> Dict[str, Any]:
        """
        Standard agent interface method.
        Expects a PatientPercept object with a `symptoms` attribute.
        """
        symptoms = getattr(percept, 'symptoms', [])
        posteriors = self.compute_posterior(symptoms)

        # Rank diagnoses by probability score
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
        """
        Generates a human-readable mathematical breakdown of the posterior logic.
        """
        symptoms_clean = [s.lower().strip().replace(' ', '_') for s in symptoms]
        likelihoods = self.likelihoods.get(disease, {})
        
        evidence = [
            f"P({s}|{disease})={likelihoods.get(s, 0.01):.2f}"
            for s in symptoms_clean
        ]
        
        prior_val = self.priors.get(disease, 0.0)
        return f"P({disease}) = {prior_val} × " + " × ".join(evidence)


# ============================================================
# MODULE TESTER / DEMO
# ============================================================
if __name__ == "__main__":
    # Test instance standalone
    bn = SimpleBayesianDiagnostics()
    
    test_symptoms = ["fever", "cough", "loss of smell", "fatigue"]
    print(f"Testing symptoms: {test_symptoms}\n")

    # 1. Test compute_posterior
    posteriors = bn.compute_posterior(test_symptoms)
    print("Ranked Diagnoses:")
    ranked = sorted(posteriors.items(), key=lambda x: x[1], reverse=True)
    for disease, prob in ranked:
        print(f"  • {disease:<15}: {prob:.2%}")

    # 2. Test explanation generator
    print("\nExplanation breakdown for top disease:")
    top_disease = ranked[0][0]
    print(" ", bn.explain(top_disease, test_symptoms))