# ============================================================
# MODULE 2: FOL Knowledge Base + Inference Engine
# Covers: Week 5 (First-Order Logic & Inference)
# ============================================================

from typing import Set, List, Dict, Tuple, Optional


class MedicalKnowledgeBase:
    """
    First-Order Logic based medical knowledge base.
    Supports forward chaining, backward chaining,
    and confidence-weighted inference.
    """

    def __init__(self):
        self.facts: Set[str] = set()
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
        ]
        for conditions, conclusion, cf in disease_rules:
            self.add_rule(conditions, conclusion, cf)

    def add_fact(self, fact: str, certainty: float = 1.0):
        self.facts.add(fact)
        self.certainty_factors[fact] = certainty

    def add_rule(self, conditions: List[str],
                 conclusion: str, certainty: float = 1.0):
        self.rules.append((conditions, conclusion, certainty))

    def load_patient_symptoms(self, symptoms: List[str]):
        """Load patient symptoms as facts"""
        for symptom in symptoms:
            self.add_fact(symptom.lower().replace(' ', '_'))

    def forward_chain(self, verbose: bool = False) -> Dict[str, float]:
        """Forward chaining with certainty factors"""
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
                    # Combine certainty factors
                    cond_cfs = [
                        self.certainty_factors.get(c,
                                                   inferred.get(c, 1.0))
                        for c in conditions
                    ]
                    combined_cf = rule_cf * min(cond_cfs)
                    inferred[conclusion] = round(combined_cf, 4)

                    if verbose:
                        cond_str = " ∧ ".join(conditions)
                        print(f"  Iter {iteration}: "
                              f"{cond_str} → {conclusion} "
                              f"(CF={combined_cf:.3f})")
                    changed = True
        return inferred

    def backward_chain(self, goal: str,
                       visited: Optional[Set] = None,
                       depth: int = 0) -> Tuple[bool, float]:
        """Backward chaining — prove a goal"""
        indent = "  " * depth
        visited = visited or set()

        if goal in self.facts:
            return True, self.certainty_factors.get(goal, 1.0)
        if goal in visited:
            return False, 0.0
        visited.add(goal)

        for conditions, conclusion, rule_cf in self.rules:
            if conclusion == goal:
                results = [
                    self.backward_chain(c, visited.copy(), depth + 1)
                    for c in conditions
                ]
                if all(proved for proved, _ in results):
                    cf = rule_cf * min(cf for _, cf in results)
                    return True, round(cf, 4)
        return False, 0.0

    def analyze(self, percept) -> Dict:
        """Module interface for the agent"""
        self.facts = set()
        self.certainty_factors = {}
        self.load_patient_symptoms(percept.symptoms)

        # Add vitals as facts
        if percept.temperature > 38.0:
            self.add_fact("fever",
                          min(1.0, (percept.temperature - 37.0) / 3.0))
        if percept.temperature > 39.5:
            self.add_fact("high_fever", 1.0)
        if percept.heart_rate > 100:
            self.add_fact("tachycardia", 1.0)

        inferred = self.forward_chain()
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

        return {
            'summary': f"Inferred {len(inferred)} conclusions -> {top_disease}",
            'diagnosis': top_disease,
            'confidence': confidence,
            'all_inferred': inferred
        }

    def get_explanation(self, diagnosis: str) -> str:
        """Explain how a diagnosis was reached"""
        for conditions, conclusion, cf in self.rules:
            if conclusion == diagnosis:
                return (f"'{diagnosis}' derived from: "
                        f"{' + '.join(conditions)} (CF={cf})")
        return f"'{diagnosis}' is a base fact"


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


if __name__ == "__main__":
    run_kb_validation()