# ============================================================
# MODULE 6: Fuzzy Logic — Patient Severity Assessment
# Covers: Week 12 (Fuzzy Logic & Fuzzy Control Systems)
# ============================================================

import numpy as np
from typing import Dict, Any


class FuzzySeverityAssessor:
    """
    Fuzzy logic system for patient severity assessment.
    Inputs:  Temperature (°C), Heart Rate (BPM), Symptom Count
    Output:  Severity Score (0-100) & Categorical Severity Label
    """

    def _membership_temp(self, temp: float) -> Dict[str, float]:
        """
        Temperature membership functions.
        Calculates degrees of membership across [normal, mild, high, critical].
        Clamped to [0.0, 1.0] to prevent overflow.
        """
        return {
            'normal': max(0.0, min(1.0, (37.5 - temp) / 1.0)) if temp <= 37.5 else 0.0,
            'mild': max(0.0, min(1.0, 1.0 - abs(temp - 38.0) / 1.0)),
            'high': max(0.0, min(1.0, 1.0 - abs(temp - 39.0) / 1.0)),
            'critical': max(0.0, min(1.0, (temp - 39.0) / 1.5)) if temp >= 39.0 else 0.0
        }

    def _membership_hr(self, hr: int) -> Dict[str, float]:
        """
        Heart rate membership functions.
        Calculates degrees of membership across [low, normal, elevated, high].
        Clamped to [0.0, 1.0].
        """
        return {
            'low': max(0.0, min(1.0, (70.0 - hr) / 10.0)) if hr <= 70 else 0.0,
            'normal': max(0.0, min(1.0, 1.0 - abs(hr - 80.0) / 20.0)),
            'elevated': max(0.0, min(1.0, 1.0 - abs(hr - 100.0) / 15.0)),
            'high': max(0.0, min(1.0, (hr - 100.0) / 20.0)) if hr >= 100 else 0.0
        }

    def _membership_symptoms(self, count: int) -> Dict[str, float]:
        """
        Symptom count membership functions.
        Calculates degrees of membership across [few, moderate, many].
        Clamped to [0.0, 1.0].
        """
        return {
            'few': max(0.0, min(1.0, (3.0 - count) / 2.0)) if count <= 3 else 0.0,
            'moderate': max(0.0, min(1.0, 1.0 - abs(count - 4.0) / 2.0)),
            'many': max(0.0, min(1.0, (count - 5.0) / 3.0)) if count >= 5 else 0.0
        }

    def _defuzzify(self, severity_rules: Dict[str, float]) -> float:
        """
        Centroid Defuzzification Method.
        Converts rule activation strengths into a crisp continuous score (0-100).
        Includes 1e-10 epsilon to prevent division by zero errors.
        """
        centers = {
            'low': 15.0,
            'mild': 35.0,
            'moderate': 55.0,
            'high': 75.0,
            'critical': 92.0
        }

        numerator = sum(centers[k] * v for k, v in severity_rules.items() if k in centers)
        denominator = sum(severity_rules.values()) + 1e-10
        return numerator / denominator

    def _classify(self, score: float) -> str:
        """Categorizes continuous 0-100 score into severity levels."""
        if score >= 80.0:
            return "CRITICAL"
        elif score >= 60.0:
            return "HIGH"
        elif score >= 40.0:
            return "MODERATE"
        elif score >= 20.0:
            return "MILD"
        return "LOW"

    def assess(self, temperature: float, heart_rate: int, symptom_count: int) -> Dict[str, Any]:
        """Full Fuzzy Inference Pipeline: Fuzzification -> Rule Evaluation -> Defuzzification"""
        # Step 1: Fuzzification
        temp_mf = self._membership_temp(temperature)
        hr_mf = self._membership_hr(heart_rate)
        symptom_mf = self._membership_symptoms(symptom_count)

        # Step 2: Rule Evaluation (AND = min, OR = max)
        # NOTE: a bug was found and fixed here. The original rules only
        # let a *critical* temperature register as severe when it was
        # ALSO combined (via min/AND) with an elevated heart rate or many
        # symptoms. A patient with e.g. temp=45C but a normal heart rate
        # and few symptoms had temp_mf['critical']=1.0 contributing to
        # NOTHING — every rule evaluated to 0, so defuzzify() divided ~0
        # by ~0 and returned severity_score=0.0 (LOW) for a reading that
        # is medically critical on its own. Fix: OR in the raw temp/hr
        # membership on its own (not just in combination), so an extreme
        # single vital reading can never be diluted away to zero just
        # because the other two inputs happen to look normal.
        rules = {
            'critical': max(
                min(temp_mf['critical'], hr_mf['high']),
                min(temp_mf['critical'], symptom_mf['many']),
                temp_mf['critical'],  # critical temp alone is enough
                hr_mf['high']  # critical heart rate alone is enough
            ),
            'high': max(
                min(temp_mf['high'], hr_mf['elevated']),
                min(temp_mf['high'], symptom_mf['many']),
                min(temp_mf['mild'], hr_mf['high']),
                temp_mf['high']  # high temp alone still counts as 'high'
            ),
            'moderate': max(
                min(temp_mf['mild'], hr_mf['normal']),
                min(temp_mf['high'], symptom_mf['moderate']),
                min(temp_mf['normal'], symptom_mf['many'])
            ),
            'mild': max(
                min(temp_mf['mild'], symptom_mf['few']),
                min(temp_mf['normal'], symptom_mf['moderate'])
            ),
            'low': min(
                temp_mf['normal'],
                hr_mf['normal'],
                symptom_mf['few']
            )
        }

        # Step 3: Defuzzification & Classification
        severity_score = self._defuzzify(rules)
        severity_label = self._classify(severity_score)

        return {
            'severity_score': round(severity_score, 2),
            'severity_label': severity_label,
            'rule_strengths': {k: round(v, 3) for k, v in rules.items()},
            'memberships': {
                'temperature': temp_mf,
                'heart_rate': hr_mf,
                'symptoms': symptom_mf
            }
        }

    def analyze(self, percept: Any) -> Dict[str, Any]:
        """
        Standard agent interface method.
        Extracts temperature, heart rate, and symptom count from the PatientPercept object.
        """
        temperature = getattr(percept, 'temperature', 37.0)
        heart_rate = getattr(percept, 'heart_rate', 75)
        symptoms = getattr(percept, 'symptoms', [])

        result = self.assess(temperature, heart_rate, len(symptoms))

        # Format required standard output keys for agent.py integration.
        #
        # NOTE — bug fixed here: this used to also set result['diagnosis']
        # = result['severity_label'] (e.g. "LOW", "HIGH"). agent.py's
        # _aggregate_diagnosis() treats *any* module's 'diagnosis' key as
        # a vote for the patient's disease — so a severity label like
        # "LOW" was literally competing against "Influenza"/"Malaria"/etc.
        # for the final diagnosis. It went unnoticed as long as the 4 real
        # diagnostic modules agreed (their 4 votes outweighed Fuzzy's 1),
        # but with ambiguous symptoms where KB/Bayes/ML/NN each guess
        # differently, Fuzzy's stray vote could tip — or even win — a tie.
        # FuzzyController assesses severity, not disease, so it should
        # never participate in that vote at all. Its severity info is
        # still returned below (severity_label/severity_score) — agent.py
        # now reads those explicitly to influence urgency instead.
        result['summary'] = f"Severity: {result['severity_label']} ({result['severity_score']:.1f}/100)"

        return result


# ============================================================
# MODULE TESTER (Matches Lab Manual Test Bench)
# ============================================================
if __name__ == "__main__":
    fa = FuzzySeverityAssessor()

    test_cases = [
        (37.0, 72, 2, "Normal patient"),
        (38.5, 95, 4, "Mild illness"),
        (39.8, 115, 7, "Severe case"),
        (40.2, 130, 9, "Critical case"),
    ]

    print("--- Testing Fuzzy Severity Assessor ---")
    for temp, hr, count, desc in test_cases:
        res = fa.assess(temp, hr, count)
        print(f"{desc:<15}: Score = {res['severity_score']:<5} | Label = {res['severity_label']}")