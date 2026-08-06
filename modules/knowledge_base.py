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
<<<<<<< HEAD
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
=======
        self.rules: List[Tuple] = []
        self.certainty_factors: Dict[str, float] = {}
        self._load_medical_knowledge()

    # Maps this KB's internal "_suspected" conclusions to the exact
    # disease-name strings used in data/patient_records.csv (and by
    # ml_classifier.py / neural_network.py / bayesian_net.py). Without
    # this mapping, the Agent's cross-module vote in _aggregate_diagnosis()
    # would never match "covid19_suspected" against "COVID-19" — they'd
    # look like two different diagnoses instead of the same one.
    SUSPECT_TO_DISEASE = {
        "covid19_suspected": "COVID-19",
        "common_cold_suspected": "Common Cold",
        "food_poisoning_suspected": "Food Poisoning",
        "hypertension_suspected": "Hypertension",
        "influenza_suspected": "Influenza",
        "malaria_suspected": "Malaria",
        "migraine_suspected": "Migraine",
        "pneumonia_suspected": "Pneumonia",
        "typhoid_suspected": "Typhoid",
    }

    def _load_medical_knowledge(self):
        """Load domain medical knowledge.

        Rules are written from the symptom patterns actually present in
        data/patient_records.csv (e.g. COVID-19 patients in that dataset
        show loss_of_smell/loss_of_taste ~80-98% of the time; Influenza
        and Malaria share fever/fatigue/body_ache/chills but Influenza
        also carries a cough while Malaria mostly doesn't).
        """
        # ── Symptom Facts (loaded dynamically per patient) ──
        # ── Disease Rules ──
        disease_rules = [
            # (conditions,                                          conclusion,                 certainty)
            (["cough", "shortness_of_breath", "loss_of_smell", "loss_of_taste"],
             "covid19_suspected", 0.90),
            (["sore_throat", "runny_nose", "sneezing", "cough"],
             "common_cold_suspected", 0.85),
            (["nausea", "vomiting", "diarrhea"],
             "food_poisoning_suspected", 0.88),
            (["headache", "dizziness", "high_blood_pressure"],
             "hypertension_suspected", 0.90),
            (["fever", "cough", "fatigue", "body_ache", "chills"],
             "influenza_suspected", 0.82),
            (["fever", "headache", "fatigue", "body_ache", "chills"],
             "malaria_suspected", 0.80),
            (["headache", "nausea", "dizziness"],
             "migraine_suspected", 0.85),
            (["fever", "cough", "chest_pain", "shortness_of_breath"],
             "pneumonia_suspected", 0.90),
            (["fever", "headache", "fatigue", "nausea", "diarrhea"],
             "typhoid_suspected", 0.85),
            # Urgency escalation rules
            (["pneumonia_suspected"],
             "URGENT_CARE", 0.90),
            (["covid19_suspected"],
             "ISOLATE_AND_TREAT", 0.90),
            (["hypertension_suspected"],
             "MONITOR_BP", 0.85),
>>>>>>> tabby/project-setup
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
<<<<<<< HEAD
                        self.certainty_factors.get(c, inferred.get(c, 1.0))
=======
                        self.certainty_factors.get(c,
                                                   inferred.get(c, 1.0))
>>>>>>> tabby/project-setup
                        for c in conditions
                    ]
                    combined_cf = rule_cf * min(cond_cfs)
                    inferred[conclusion] = round(combined_cf, 4)

                    if verbose:
                        cond_str = " AND ".join(conditions)
                        print(f"  Iter {iteration}: {cond_str} -> {conclusion} (CF={combined_cf:.3f})")
                    changed = True
        return inferred

<<<<<<< HEAD
    def backward_chain(
        self, goal: str, visited: Optional[Set[str]] = None, depth: int = 0
    ) -> Tuple[bool, float]:
        """Backward chaining algorithm to verify specific diagnostic goals."""
        clean_goal = goal.strip().lower().replace(' ', '_') if goal not in ["EMERGENCY", "ISOLATE_AND_TREAT", "REST_AND_MEDICATE"] else goal
=======
    def backward_chain(self, goal: str,
                       visited: Optional[Set] = None,
                       depth: int = 0) -> Tuple[bool, float]:
        """Backward chaining — prove a goal"""
        indent = "  " * depth
>>>>>>> tabby/project-setup
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
<<<<<<< HEAD
        if temp > 38.0:
            self.add_fact("fever", min(1.0, (temp - 37.0) / 3.0))
        if temp > 39.5:
=======
        if percept.temperature > 38.0:
            self.add_fact("fever",
                          min(1.0, (percept.temperature - 37.0) / 3.0))
        if percept.temperature > 39.5:
>>>>>>> tabby/project-setup
            self.add_fact("high_fever", 1.0)
        if heart_rate > 100:
            self.add_fact("tachycardia", 1.0)

        inferred = self.forward_chain()
<<<<<<< HEAD
        diseases = {
            k: v for k, v in inferred.items()
            if 'suspected' in k or 'confirmed' in k
        }
=======
        diseases = {k: v for k, v in inferred.items()
                    if k in self.SUSPECT_TO_DISEASE}

        if diseases:
            top_suspect = max(diseases, key=diseases.get)
            top_disease = self.SUSPECT_TO_DISEASE[top_suspect]
            confidence = diseases[top_suspect]
        else:
            # No disease rule fired — nothing here suggests illness
            top_disease = "Healthy"
            confidence = 0.5
>>>>>>> tabby/project-setup

        return {
<<<<<<< HEAD
            'summary': f"Inferred {len(inferred)} conclusions",
            'diagnosis': top,
            'confidence': diseases.get(top, 0.5),
=======
            'summary': f"Inferred {len(inferred)} conclusions -> {top_disease}",
            'diagnosis': top_disease,
            'confidence': confidence,
>>>>>>> tabby/project-setup
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


<<<<<<< HEAD
=======
# ============================================================
# VALIDATION TEST SCRIPT FOR KNOWLEDGE BASE
# ============================================================
import sys
import io
from dataclasses import dataclass

# Force UTF-8 encoding for Windows terminals to support symbols like checkmarks
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# Mocking the PatientPercept from agent.py for standalone testing
@dataclass
class MockPatientPercept:
    patient_id: str
    symptoms: list
    temperature: float
    heart_rate: int


def run_kb_validation():
    print("--- STARTING KNOWLEDGE BASE VALIDATION ---")

    kb = MedicalKnowledgeBase()

    # 1. Test Fact Loading & Base Rules
    print("\n[✓] 1. Testing Fact Loading...")
    kb.load_patient_symptoms(["cough", "shortness of breath", "loss of smell", "loss of taste"])
    kb.add_fact("fever", 0.9)  # Adding fever manually with a certainty factor
    print(f"Facts successfully loaded: {kb.facts}")

    # 2. Test Forward Chaining (Data-Driven)
    print("\n[✓] 2. Testing Forward Chaining (Verbose)...")
    inferred = kb.forward_chain(verbose=True)
    print(f"\nFinal Inferred Conclusions: {inferred}")

    # 3. Test Backward Chaining (Goal-Driven)
    print("\n[✓] 3. Testing Backward Chaining...")
    # Checking if the system can prove a specific goal working backwards
    proved_covid, cf_covid = kb.backward_chain("covid19_suspected")
    print(f"Goal 'covid19_suspected' proven? {proved_covid} (CF: {cf_covid})")

    # Checking a false goal
    proved_dengue, cf_dengue = kb.backward_chain("dengue_suspected")
    print(f"Goal 'dengue_suspected' proven? {proved_dengue} (CF: {cf_dengue})")

    # 4. Test Explanations
    print("\n[✓] 4. Testing Diagnosis Explanations...")
    explanation = kb.get_explanation("covid19_suspected")
    print(f"Explanation Engine Output: {explanation}")

    # 5. Test Integration with Agent (analyze method)
    print("\n[✓] 5. Testing Agent Interface (analyze method)...")
    # This patient data should trigger the pneumonia rule
    test_patient = MockPatientPercept(
        patient_id="PT-123",
        symptoms=["cough", "chest_pain", "shortness_of_breath"],
        temperature=39.8,  # This will automatically add 'fever' and 'high_fever' facts
        heart_rate=105  # This adds 'tachycardia'
    )

    agent_result = kb.analyze(test_patient)
    print(f"Analyze Method Result:\n{agent_result}")


>>>>>>> tabby/project-setup
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