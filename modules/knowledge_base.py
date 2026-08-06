# ============================================================
# MODULE 2: FOL Knowledge Base + Inference Engine
# Covers: Week 5 (First-Order Logic & Inference)
# ============================================================

from typing import Set, List, Dict, Tuple, Optional, Any


class MedicalKnowledgeBase:
    """
    First-Order Logic based medical knowledge base.
    Supports forward chaining, backward chaining,
    and confidence-weighted inference.
    """

    def __init__(self):
        self.facts: Set[str] = set()
        self.rules: List[Tuple[List[str], str, float]] = []
        self.certainty_factors: Dict[str, float] = {}
        self._load_medical_knowledge()

    def _load_medical_knowledge(self) -> None:
        """Load domain medical knowledge rules."""
        disease_rules = [
            # (conditions, conclusion, certainty)
            (["fever", "cough", "fatigue"], "flu_suspected", 0.75),
            (["fever", "cough", "loss_of_smell", "fatigue"], "covid19_suspected", 0.85),
            (["fever", "rash", "joint_pain"], "dengue_suspected", 0.80),
            (["chest_pain", "shortness_of_breath", "sweating"], "cardiac_event_suspected", 0.90),
            (["headache", "stiff_neck", "high_fever", "light_sensitivity"], "meningitis_suspected", 0.88),
            (["cough", "weight_loss", "night_sweats", "fatigue"], "tuberculosis_suspected", 0.82),
            (["frequent_urination", "excessive_thirst", "blurred_vision"], "diabetes_suspected", 0.78),
            (["flu_suspected", "high_fever"], "flu_confirmed", 0.85),
            (["covid19_suspected", "positive_pcr"], "covid19_confirmed", 0.99),
            (["cardiac_event_suspected", "elevated_troponin"], "myocardial_infarction", 0.95),
            # Urgency rules
            (["myocardial_infarction"], "EMERGENCY", 1.00),
            (["meningitis_suspected"], "EMERGENCY", 0.95),
            (["covid19_confirmed"], "ISOLATE_AND_TREAT", 0.99),
            (["flu_confirmed"], "REST_AND_MEDICATE", 0.90),
        ]
        for conditions, conclusion, cf in disease_rules:
            self.add_rule(conditions, conclusion, cf)

    def add_fact(self, fact: str, certainty: float = 1.0) -> None:
        """Adds a fact and its certainty factor to the KB."""
        clean_fact = fact.strip().lower().replace(' ', '_')
        self.facts.add(clean_fact)
        self.certainty_factors[clean_fact] = max(
            self.certainty_factors.get(clean_fact, 0.0), certainty
        )

    def add_rule(self, conditions: List[str], conclusion: str, certainty: float = 1.0) -> None:
        """Adds an FOL rule to the inference engine."""
        clean_conditions = [c.strip().lower().replace(' ', '_') for c in conditions]
        clean_conclusion = conclusion.strip().lower().replace(' ', '_') if conclusion not in ["EMERGENCY", "ISOLATE_AND_TREAT", "REST_AND_MEDICATE"] else conclusion
        self.rules.append((clean_conditions, clean_conclusion, certainty))

    def load_patient_symptoms(self, symptoms: List[str]) -> None:
        """Load patient symptoms as facts."""
        for symptom in symptoms:
            self.add_fact(symptom)

    def forward_chain(self, verbose: bool = False) -> Dict[str, float]:
        """Forward chaining algorithm using certainty factors."""
        inferred = {}
        changed = True
        iteration = 0

        while changed:
            changed = False
            iteration += 1
            for conditions, conclusion, rule_cf in self.rules:
                all_known = all(
                    c in self.facts or c in inferred for c in conditions
                )
                if all_known and conclusion not in inferred:
                    cond_cfs = [
                        self.certainty_factors.get(c, inferred.get(c, 1.0))
                        for c in conditions
                    ]
                    combined_cf = rule_cf * min(cond_cfs)
                    inferred[conclusion] = round(combined_cf, 4)

                    if verbose:
                        cond_str = " AND ".join(conditions)
                        print(f"  Iter {iteration}: {cond_str} -> {conclusion} (CF={combined_cf:.3f})")
                    changed = True
        return inferred

    def backward_chain(
        self, goal: str, visited: Optional[Set[str]] = None, depth: int = 0
    ) -> Tuple[bool, float]:
        """Backward chaining algorithm to verify specific diagnostic goals."""
        clean_goal = goal.strip().lower().replace(' ', '_') if goal not in ["EMERGENCY", "ISOLATE_AND_TREAT", "REST_AND_MEDICATE"] else goal
        visited = visited or set()

        if clean_goal in self.facts:
            return True, self.certainty_factors.get(clean_goal, 1.0)
        if clean_goal in visited:
            return False, 0.0
        visited.add(clean_goal)

        for conditions, conclusion, rule_cf in self.rules:
            if conclusion == clean_goal:
                results = [
                    self.backward_chain(c, visited.copy(), depth + 1)
                    for c in conditions
                ]
                if all(proved for proved, _ in results):
                    cf = rule_cf * min(cf for _, cf in results)
                    return True, round(cf, 4)
        return False, 0.0

    def analyze(self, percept: Any) -> Dict[str, Any]:
        """Standard module interface called by HealthcareDiagnosticAgent."""
        self.facts = set()
        self.certainty_factors = {}

        # Safely extract attributes whether percept is an object or dictionary
        symptoms = getattr(percept, 'symptoms', []) if hasattr(percept, 'symptoms') else percept.get('symptoms', [])
        temp = getattr(percept, 'temperature', 37.0) if hasattr(percept, 'temperature') else percept.get('temperature', 37.0)
        heart_rate = getattr(percept, 'heart_rate', 70) if hasattr(percept, 'heart_rate') else percept.get('heart_rate', 70)

        self.load_patient_symptoms(symptoms)

        # Add vitals as facts
        if temp > 38.0:
            self.add_fact("fever", min(1.0, (temp - 37.0) / 3.0))
        if temp > 39.5:
            self.add_fact("high_fever", 1.0)
        if heart_rate > 100:
            self.add_fact("tachycardia", 1.0)

        inferred = self.forward_chain()
        diseases = {
            k: v for k, v in inferred.items()
            if 'suspected' in k or 'confirmed' in k
        }

        top = max(diseases, key=diseases.get) if diseases else "Unknown"
        return {
            'summary': f"Inferred {len(inferred)} conclusions",
            'diagnosis': top,
            'confidence': diseases.get(top, 0.5),
            'all_inferred': inferred
        }

    def get_explanation(self, diagnosis: str) -> str:
        """Explains how a diagnosis was reached."""
        clean_diag = diagnosis.strip().lower().replace(' ', '_')
        for conditions, conclusion, cf in self.rules:
            if conclusion == clean_diag:
                return (
                    f"'{clean_diag}' derived from: "
                    f"{' + '.join(conditions)} (CF={cf})"
                )
        return f"'{diagnosis}' is a base fact or unproved."


if __name__ == "__main__":
    # --- Standalone Test Execution ---
    kb = MedicalKnowledgeBase()

    # Create dummy patient object matching PatientPercept
    class DummyPercept:
        patient_id = "P-101"
        symptoms = ["cough", "loss of smell", "fatigue"]
        temperature = 38.9
        heart_rate = 105

    dummy_patient = DummyPercept()

    print("\n=== KNOWLEDGE BASE STANDALONE TEST ===")
    
    # 1. Test standard analyze interface
    result = kb.analyze(dummy_patient)
    print(f"Summary:       {result['summary']}")
    print(f"Top Diagnosis: {result['diagnosis']}")
    print(f"Confidence:    {result['confidence'] * 100:.1f}%")
    print("All Inferred Facts:", result['all_inferred'])

    # 2. Test Explanation Generator
    explanation = kb.get_explanation(result['diagnosis'])
    print(f"\nExplanation: {explanation}")

    # 3. Test Backward Chaining
    proved, cf = kb.backward_chain("covid19_suspected")
    print(f"\nBackward Chaining 'covid19_suspected': Proved={proved}, Confidence={cf}")