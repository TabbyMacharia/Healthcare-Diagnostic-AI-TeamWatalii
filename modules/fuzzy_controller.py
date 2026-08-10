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
        Ensures proper open-ended boundary clipping for extreme values.
        """
        return {
            'normal': 1.0 if temp <= 36.5 else max(0.0, min(1.0, (37.5 - temp) / 1.0)),
            'mild': max(0.0, min(1.0, 1.0 - abs(temp - 38.0) / 0.8)),
            'high': max(0.0, min(1.0, 1.0 - abs(temp - 39.0) / 0.8)),
            'critical': 0.0 if temp < 39.0 else min(1.0, (temp - 39.0) / 1.5)
        }

    def _membership_hr(self, hr: int) -> Dict[str, float]:
        """
        Heart rate membership functions.
        Calculates degrees of membership across [low, normal, elevated, high].
        Ensures lower and upper bounds clip to 1.0 at physiological extremes.
        """
        return {
            'low': 1.0 if hr <= 60 else max(0.0, min(1.0, (75.0 - hr) / 15.0)),
            'normal': max(0.0, min(1.0, 1.0 - abs(hr - 75.0) / 15.0)),
            'elevated': max(0.0, min(1.0, 1.0 - abs(hr - 95.0) / 15.0)),
            'high': 0.0 if hr < 95 else min(1.0, (hr - 95.0) / 25.0)
        }

    def _membership_symptoms(self, count: int) -> Dict[str, float]:
        """
        Symptom count membership functions.
        Calculates degrees of membership across [few, moderate, many].
        """
        return {
            'few': 1.0 if count <= 1 else max(0.0, min(1.0, (3.0 - count) / 2.0)),
            'moderate': max(0.0, min(1.0, 1.0 - abs(count - 3.5) / 1.5)),
            'many': 0.0 if count < 4 else min(1.0, (count - 4.0) / 2.0)
        }

    def _defuzzify(self, severity_rules: Dict[str, float]) -> float:
        """
        Centroid Defuzzification Method.
        Converts rule activation strengths into a crisp continuous score (0-100).
        Includes safety handling to prevent 0.0 outputs when total weight is zero.
        """
        centers = {
            'low': 15.0,
            'mild': 35.0,
            'moderate': 55.0,
            'high': 75.0,
            'critical': 92.0
        }

        total_weight = sum(severity_rules.values())

        # Fallback safeguard: If no rules fired or total weight is zero, return neutral score
        if total_weight == 0.0:
            return 50.0

        numerator = sum(centers[k] * v for k, v in severity_rules.items() if k in centers)
        return numerator / total_weight

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
        rules = {
            'critical': max(
                temp_mf['critical'],
                min(temp_mf['high'], hr_mf['high']),
                min(temp_mf['critical'], symptom_mf['many'])
            ),
            'high': max(
                min(temp_mf['high'], hr_mf['elevated']),
                min(temp_mf['high'], symptom_mf['many']),
                min(temp_mf['mild'], hr_mf['high']),
                min(hr_mf['high'], symptom_mf['moderate'])
            ),
            'moderate': max(
                min(temp_mf['mild'], hr_mf['normal']),
                min(temp_mf['normal'], hr_mf['elevated']),
                min(temp_mf['high'], symptom_mf['few']),
                min(temp_mf['normal'], symptom_mf['many']),
                min(hr_mf['elevated'], symptom_mf['moderate'])
            ),
            'mild': max(
                min(temp_mf['mild'], symptom_mf['few']),
                min(temp_mf['normal'], symptom_mf['moderate']),
                min(temp_mf['normal'], hr_mf['elevated'])
            ),
            'low': max(
                min(temp_mf['normal'], hr_mf['normal'], symptom_mf['few']),
                min(temp_mf['normal'], hr_mf['low'], symptom_mf['few'])
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

        # Format required standard output keys for agent.py integration
        result['summary'] = f"Severity: {result['severity_label']} ({result['severity_score']:.1f}/100)"
        result['diagnosis'] = result['severity_label']
        result['confidence'] = round(result['severity_score'] / 100.0, 4)

        return result


# ============================================================
# MODULE TESTER
# ============================================================
if __name__ == "__main__":
    fa = FuzzySeverityAssessor()

    test_cases = [
        (37.0, 72, 2, "Normal patient"),
        (38.5, 95, 4, "Mild illness"),
        (39.8, 115, 7, "Severe case"),
        (40.2, 130, 9, "Critical case"),
        (65.0, 72, 3, "Extreme Outlier Input"),
    ]

    print("--- Testing Fuzzy Severity Assessor ---")
    for temp, hr, count, desc in test_cases:
        res = fa.assess(temp, hr, count)
        print(f"{desc:<22}: Score = {res['severity_score']:<5} | Label = {res['severity_label']}")