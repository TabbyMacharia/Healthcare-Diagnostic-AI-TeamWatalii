# ============================================================
# MODULE 1: Intelligent Agent — Healthcare Diagnostic Agent
# Covers: Week 2 (Intelligent Agents) + PEAS Framework
# ============================================================

import sys
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import datetime

# Force UTF-8 output for Windows terminals to support emojis.
# NOTE: the previous version did
#   sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# which crashes with "I/O operation on closed file" whenever stdout isn't
# a plain buffered stream (piped output, Jupyter, some IDEs, or simply
# importing this module a second time in the same process). reconfigure()
# is the standard, safe way to do the same thing in Python 3.7+, and it
# no-ops safely if stdout doesn't support it (e.g. some redirected streams).
try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass


class AgentState(Enum):
    IDLE = "idle"
    COLLECTING = "collecting_symptoms"
    DIAGNOSING = "diagnosing"
    RECOMMENDING = "recommending"
    PLANNING = "planning_treatment"
    DONE = "done"


@dataclass
class PatientPercept:
    """What the agent perceives from the environment"""
    patient_id: str
    symptoms: List[str]
    age: int
    temperature: float
    heart_rate: int
    blood_pressure: str
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().isoformat())


@dataclass
class AgentMemory:
    """Internal model — makes this a model-based agent"""
    patient_history: List[Dict] = field(default_factory=list)
    current_patient: Optional[PatientPercept] = None
    diagnosis_history: List[str] = field(default_factory=list)
    action_log: List[str] = field(default_factory=list)


class HealthcareDiagnosticAgent:
    """
    PEAS Definition:
    ─────────────────────────────────────────────────
    Performance : Diagnostic accuracy, patient safety,
                  recommendation quality, response time
    Environment : Hospital/clinic, patient data, EMR
    Actuators   : Diagnosis report, treatment plan,
                  referral recommendation, alerts
    Sensors     : Symptom input, vitals, lab results,
                  patient history
    ─────────────────────────────────────────────────
    Agent Type  : Model-Based + Goal-Based + Learning
    """

    def __init__(self):
        self.state = AgentState.IDLE
        self.memory = AgentMemory()
        self.performance_score = 0
        self._modules = {}  # Will hold sub-modules

    def register_module(self, name: str, module):
        """Plug in AI sub-modules (KB, Bayes, ML, etc.)"""
        self._modules[name] = module
        print(f"  🔌 Module registered: [{name}]")

    def perceive(self, percept: PatientPercept):
        """Step 1: Perceive the environment"""
        self.memory.current_patient = percept
        self.memory.patient_history.append({
            'id': percept.patient_id,
            'symptoms': percept.symptoms,
            'time': percept.timestamp
        })
        self.state = AgentState.COLLECTING
        self._log(f"Perceived patient {percept.patient_id} "
                  f"with {len(percept.symptoms)} symptoms")
        return self

    def think(self):
        """Step 2: Process and reason"""
        self.state = AgentState.DIAGNOSING
        self._log("Agent thinking: running diagnostic modules...")

        results = {}

        # Run each registered module
        for module_name, module in self._modules.items():
            if hasattr(module, 'analyze'):
                result = module.analyze(self.memory.current_patient)
                results[module_name] = result
                self._log(f"  [{module_name}] → {result.get('summary', 'done')}")

        self.memory.diagnosis_history.append(results)
        self.state = AgentState.RECOMMENDING
        return results

    def act(self, diagnosis_results: Dict) -> Dict:
        """Step 3: Generate action/recommendation"""
        self.state = AgentState.PLANNING
        patient = self.memory.current_patient

        # Aggregate confidence from multiple modules
        confidences = [
            v.get('confidence', 0)
            for v in diagnosis_results.values()
            if isinstance(v, dict) and 'confidence' in v
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        # Determine urgency from raw vitals/confidence...
        urgency = self._assess_urgency(patient, avg_confidence)

        # ...then fold in FuzzyController's severity assessment (Module 6),
        # taking whichever of the two is more severe. Previously Fuzzy's
        # output was computed but never actually used here — its severity
        # score only leaked into the system by incorrectly posing as a
        # disease "diagnosis" vote (now fixed in fuzzy_controller.py).
        # This is the real integration point the lab manual's architecture
        # diagram implies: Module 6 (Fuzzy Severity) feeding into the
        # final urgency/output, not just existing in isolation.
        fuzzy_result = diagnosis_results.get('FuzzyController')
        if isinstance(fuzzy_result, dict) and 'severity_label' in fuzzy_result:
            fuzzy_urgency = self._severity_to_urgency(fuzzy_result['severity_label'])
            urgency = self._more_severe(urgency, fuzzy_urgency)

        action_report = {
            'patient_id': patient.patient_id,
            'timestamp': patient.timestamp,
            'symptoms': patient.symptoms,
            'diagnosis': self._aggregate_diagnosis(diagnosis_results),
            'confidence': round(avg_confidence, 3),
            'urgency': urgency,
            'recommendations': self._generate_recommendations(
                urgency, diagnosis_results),
            'next_action': self._decide_next_action(urgency)
        }

        self.performance_score += (10 if avg_confidence > 0.7 else 5)
        self.state = AgentState.DONE
        self._log(f"Action generated: {urgency} urgency")
        return action_report

    def run(self, percept: PatientPercept) -> Dict:
        """Full agent cycle: Perceive → Think → Act"""
        self.perceive(percept)
        results = self.think()
        return self.act(results)

    _URGENCY_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def _severity_to_urgency(self, severity_label: str) -> str:
        """Maps FuzzyController's 5-level severity scale (LOW/MILD/
        MODERATE/HIGH/CRITICAL) onto the Agent's 4-level urgency scale
        (LOW/MEDIUM/HIGH/CRITICAL)."""
        return {
            "LOW": "LOW", "MILD": "LOW", "MODERATE": "MEDIUM",
            "HIGH": "HIGH", "CRITICAL": "CRITICAL",
        }.get(severity_label, "LOW")

    def _more_severe(self, a: str, b: str) -> str:
        ia = self._URGENCY_ORDER.index(a) if a in self._URGENCY_ORDER else 0
        ib = self._URGENCY_ORDER.index(b) if b in self._URGENCY_ORDER else 0
        return a if ia >= ib else b

    def _assess_urgency(self, patient, confidence):
        if patient.temperature > 39.5 or patient.heart_rate > 120:
            return "CRITICAL"
        elif patient.temperature > 38.5 or confidence > 0.8:
            return "HIGH"
        elif patient.temperature > 37.5:
            return "MEDIUM"
        return "LOW"

    def _aggregate_diagnosis(self, results):
        diagnoses = [
            v.get('diagnosis', 'Unknown')
            for v in results.values()
            if isinstance(v, dict) and 'diagnosis' in v
        ]
        if not diagnoses:
            return "Insufficient data"
        from collections import Counter
        return Counter(diagnoses).most_common(1)[0][0]

    def _generate_recommendations(self, urgency, results):
        base = {
            "CRITICAL": [
                "Immediate emergency consultation required",
                "Alert attending physician now",
                "Transfer to emergency ward",
                "Administer first-line medications"
            ],
            "HIGH": [
                " Schedule urgent appointment within 24 hours",
                "Order blood panel and cultures",
                "Prescribe symptomatic relief",
                "Monitor vitals every 2 hours"
            ],
            "MEDIUM": [
                "Schedule appointment within 3 days",
                "Over-the-counter treatment advised",
                "Monitor temperature twice daily",
                "Increase fluid intake"
            ],
            "LOW": [
                "Home rest recommended",
                "Stay hydrated",
                "Follow up if symptoms worsen",
                "General wellness monitoring"
            ]
        }
        return base.get(urgency, base["LOW"])

    def _decide_next_action(self, urgency):
        actions = {
            "CRITICAL": "EMERGENCY_REFERRAL",
            "HIGH": "URGENT_APPOINTMENT",
            "MEDIUM": "SCHEDULE_FOLLOWUP",
            "LOW": "MONITOR_AT_HOME"
        }
        return actions.get(urgency, "MONITOR_AT_HOME")

    def _log(self, message):
        entry = f"[{self.state.value}] {message}"
        self.memory.action_log.append(entry)

    def print_log(self):
        print("\n Agent Action Log:")
        print("─" * 50)
        for entry in self.memory.action_log:
            print(f"  {entry}")

    def get_performance(self):
        return {
            'total_patients': len(self.memory.patient_history),
            'performance_score': self.performance_score,
            'diagnoses_made': len(self.memory.diagnosis_history)
        }


# ============================================================
# VALIDATION TEST SCRIPT
# ============================================================

def run_agent_validation():
    print("--- STARTING AGENT VALIDATION ---")

    # 1. Instantiate Agent
    agent = HealthcareDiagnosticAgent()
    print(f"Initial State: {agent.state.name}")

    # 2. Test Module Registration
    class MockBayesianModule:
        def analyze(self, patient):
            return {
                "diagnosis": "Bacterial Infection",
                "confidence": 0.88,
                "summary": "High temp and heart rate suggest bacterial origin."
            }

    agent.register_module("Bayesian_Network", MockBayesianModule())

    # 3. Create a Patient Percept
    test_patient = PatientPercept(
        patient_id="PT-9942",
        symptoms=["Severe Headache", "Nausea"],
        age=34,
        temperature=38.9,  # Should trigger HIGH urgency
        heart_rate=95,
        blood_pressure="130/85"
    )

    # 4. Test Perceive()
    print("\n--- Testing perceive() ---")
    agent.perceive(test_patient)
    print(f"State after perceive: {agent.state.name}")
    print(f"Memory updated: {agent.memory.current_patient.patient_id == 'PT-9942'}")

    # 5. Test Think()
    print("\n--- Testing think() ---")
    results = agent.think()
    print(f"State after think: {agent.state.name}")
    print(f"Diagnosis Results: {results}")

    # 6. Test Act()
    print("\n--- Testing act() ---")
    action_report = agent.act(results)
    print(f"State after act: {agent.state.name}")
    print(f"Action Report Urgency: {action_report['urgency']}")

    # 7. Verify Logging and Performance
    agent.print_log()
    print("\nPerformance Data:", agent.get_performance())


if __name__ == "__main__":
    run_agent_validation()