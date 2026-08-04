# ============================================================
# CAPSTONE MAIN APPLICATION
# Intelligent Healthcare Diagnostic Assistant
# Reconstructed Application
# ============================================================

import warnings

warnings.filterwarnings("ignore")

from modules.agent import (
    HealthcareDiagnosticAgent,
    PatientPercept
)

print("Loading agent...")
from modules.agent import HealthcareDiagnosticAgent, PatientPercept

print("Loading knowledge base...")
from modules.knowledge_base import MedicalKnowledgeBase

print("Loading bayesian network...")
from modules.bayesian_net import SimpleBayesianDiagnostics

print("Loading ML classifier...")
from modules.ml_classifier import MLDiagnosticClassifier

print("Loading neural network...")
from modules.neural_network import NeuralDiagnosticModel

print("Loading fuzzy controller...")
from modules.fuzzy_controller import FuzzySeverityAssessor

print("Loading planner...")
from modules.planner import TreatmentPlanner

print("All modules loaded!")


# ============================================================
# Console Colours
# ============================================================

class C:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


# ============================================================
# Banner
# ============================================================

def banner():
    print(f"""
{C.BLUE}{C.BOLD}
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║      🏥 INTELLIGENT HEALTHCARE DIAGNOSTIC AI SYSTEM          ║
║                                                              ║
║        Artificial Intelligence Capstone Project              ║
║                                                              ║
║  Intelligent Agent                                           ║
║  Knowledge Base (FOL)                                        ║
║  Bayesian Network                                            ║
║  Machine Learning                                            ║
║  Deep Neural Network                                         ║
║  Fuzzy Logic                                                 ║
║  AI Planning                                                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
{C.END}
""")


# ============================================================
# Build AI System
# ============================================================

def build_system():

    print(f"{C.YELLOW}Initializing AI Modules...{C.END}\n")

    agent = HealthcareDiagnosticAgent()

    modules = {
        "KnowledgeBase": MedicalKnowledgeBase(),
        "BayesianNet": SimpleBayesianDiagnostics(),
        "MLClassifier": MLDiagnosticClassifier(),
        "NeuralNetwork": NeuralDiagnosticModel(),
        "FuzzyController": FuzzySeverityAssessor(),
        "TreatmentPlanner": TreatmentPlanner()
    }

    for name, module in modules.items():
        print(f"Loading {name}...")
        agent.register_module(name, module)

    print(f"\n{C.GREEN}All AI modules successfully loaded.{C.END}\n")

    return agent


# ============================================================
# Patient Input
# ============================================================

def get_patient():

    print("=" * 60)
    print("PATIENT INFORMATION")
    print("=" * 60)

    patient_id = input("Patient ID: ").strip()

    age = int(input("Age: "))

    temperature = float(input("Temperature (°C): "))

    heart_rate = int(input("Heart Rate (bpm): "))

    blood_pressure = input("Blood Pressure (e.g. 120/80): ").strip()

    print()

    print("Enter symptoms separated by commas")

    print("Example:")
    print("fever,cough,fatigue,headache")

    symptoms = input("\nSymptoms: ")

    symptom_list = [
        s.strip().lower().replace(" ", "_")
        for s in symptoms.split(",")
        if s.strip()
    ]

    return PatientPercept(
        patient_id=patient_id,
        symptoms=symptom_list,
        age=age,
        temperature=temperature,
        heart_rate=heart_rate,
        blood_pressure=blood_pressure
    )


# ============================================================
# Display Module Results
# ============================================================

def show_module_results(results):

    print()
    print("=" * 60)
    print("INDIVIDUAL AI MODULE RESULTS")
    print("=" * 60)

    for module_name, result in results.items():

        print(f"\n[{module_name}]")

        if isinstance(result, dict):

            diagnosis = result.get("diagnosis", "Unknown")
            confidence = result.get("confidence", 0)
            summary = result.get("summary", "")

            print(f"Diagnosis : {diagnosis}")
            print(f"Confidence: {confidence:.2%}")

            if summary:
                print(f"Summary   : {summary}")
                # ============================================================
# Display Final Report
# ============================================================

def show_final_report(report):

    print()
    print("=" * 60)
    print("FINAL DIAGNOSIS REPORT")
    print("=" * 60)

    print(f"Patient ID : {report['patient_id']}")
    print(f"Diagnosis  : {report['diagnosis']}")
    print(f"Confidence : {report['confidence']:.2%}")
    print(f"Urgency    : {report['urgency']}")
    print(f"Next Action: {report['next_action']}")

    print("\nSymptoms:")
    for symptom in report["symptoms"]:
        print(f"  • {symptom}")

    print("\nRecommendations:")

    for recommendation in report["recommendations"]:
        print(f"  ✓ {recommendation}")


# ============================================================
# Treatment Planner
# ============================================================

def show_treatment_plan(agent, report):

    print()
    print("=" * 60)
    print("AI TREATMENT PLAN")
    print("=" * 60)

    planner = agent._modules.get("TreatmentPlanner")

    if planner is None:
        print("Treatment planner not available.")
        return

    plan = planner.create_treatment_plan(
        report["diagnosis"],
        report["urgency"]
    )

    if "error" in plan:
        print(plan["error"])
        return

    print(f"Diagnosis : {plan['diagnosis']}")
    print(f"Urgency   : {plan['urgency']}")
    print(f"Steps     : {plan['steps']}")
    print(f"Duration  : {plan['total_duration']}")

    print("\nTreatment Steps")

    for step in plan["plan"]:
        print(
            f"{step['step']}. "
            f"{step['action']} "
            f"({step['duration']})"
        )


# ============================================================
# Performance
# ============================================================

def show_performance(agent):

    perf = agent.get_performance()

    print()
    print("=" * 60)
    print("AGENT PERFORMANCE")
    print("=" * 60)

    print(f"Patients Diagnosed : {perf['total_patients']}")
    print(f"Performance Score  : {perf['performance_score']}")
    print(f"Diagnoses Made     : {perf['diagnoses_made']}")


# ============================================================
# Agent Log
# ============================================================

def show_agent_log(agent):

    print()
    print("=" * 60)
    print("AGENT LOG")
    print("=" * 60)

    agent.print_log()


# ============================================================
# Run Diagnosis
# ============================================================

def run_diagnosis(agent):

    patient = get_patient()

    print("\nRunning Intelligent Agent...")
    agent.perceive(patient)

    print("Running AI Modules...\n")

    module_results = agent.think()

    report = agent.act(module_results)

    show_module_results(module_results)

    show_final_report(report)

    show_treatment_plan(agent, report)

    show_performance(agent)

    show_agent_log(agent)

    return report


# ============================================================
# Main Menu
# ============================================================

def menu():

    banner()

    agent = build_system()

    while True:

        print()
        print("=" * 60)
        print("MAIN MENU")
        print("=" * 60)

        print("1. Diagnose Patient")
        print("2. Exit")

        choice = input("\nSelect option: ").strip()

        if choice == "1":

            try:
                run_diagnosis(agent)

            except KeyboardInterrupt:
                print("\nOperation cancelled.")

            except Exception as e:
                print(f"\nError: {e}")

        elif choice == "2":

            print("\nThank you for using the Healthcare Diagnostic AI.")
            break

        else:

            print("Invalid choice. Please try again.")
            # ============================================================
# Program Entry Point
# ============================================================

if __name__ == "__main__":
    menu()