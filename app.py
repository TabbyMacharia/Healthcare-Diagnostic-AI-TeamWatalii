# ============================================================  
# CAPSTONE MAIN APPLICATION  
# Intelligent Healthcare Diagnostic Assistant  
# Introduction to AI — 13-Week Capstone  
# ============================================================  

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow GPU/CUDA logs
import sys  
import json  
import warnings  
import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib.gridspec as gridspec  
warnings.filterwarnings('ignore')

# Import core modules
from modules.agent          import HealthcareDiagnosticAgent, PatientPercept  
from modules.knowledge_base import MedicalKnowledgeBase  
from modules.ml_classifier  import MLDiagnosticClassifier  
from modules.neural_network import NeuralDiagnosticModel  
from modules.planner        import TreatmentPlanner  

# Safely handle optional/extra modules if present
try:
    from modules.bayesian_net import SimpleBayesianDiagnostics  
except ImportError:
    SimpleBayesianDiagnostics = None

try:
    from modules.fuzzy_controller import FuzzySeverityAssessor  
except ImportError:
    FuzzySeverityAssessor = None


# ── ANSI Colors ────────────────────────────────────────────  
class C:  
    HEADER = '\033[95m'; BLUE   = '\033[94m'  
    GREEN  = '\033[92m'; YELLOW = '\033[93m'  
    RED    = '\033[91m'; BOLD   = '\033[1m'  
    END    = '\033[0m'  


def banner():  
    print(f"""  
{C.BOLD}{C.BLUE}  
╔══════════════════════════════════════════════════════════╗  
║        🏥 INTELLIGENT HEALTHCARE DIAGNOSTIC AI           ║  
║         Introduction to AI — Capstone Project            ║  
║  Modules: Agents | Logic | Bayes | ML | DNN | Planner    ║  
╚══════════════════════════════════════════════════════════╝  
{C.END}""")  


def section(title: str):  
    print(f"\n{C.BOLD}{C.YELLOW}{'═'*60}{C.END}")  
    print(f"{C.BOLD}{C.YELLOW}  {title}{C.END}")  
    print(f"{C.BOLD}{C.YELLOW}{'═'*60}{C.END}")  


def build_system() -> HealthcareDiagnosticAgent:  
    """Instantiate and wire all AI modules into the master agent."""  
    section("🔧 Building AI System — Registering Modules")  

    agent = HealthcareDiagnosticAgent()  

    print("\n  Initializing sub-modules...")  

    # 1. First-Order Logic Knowledge Base
    kb = MedicalKnowledgeBase()
    agent.register_module("knowledge_base", kb)

    # 2. Bayesian Diagnostics (Optional)
    if SimpleBayesianDiagnostics:
        bayesian = SimpleBayesianDiagnostics()
        agent.register_module("bayesian_net", bayesian)

    # 3. Supervised Machine Learning Classifier
    ml_clf = MLDiagnosticClassifier()
    ml_clf.train(verbose=False)
    agent.register_module("ml_classifier", ml_clf)

    # 4. Deep Neural Network
    nn_model = NeuralDiagnosticModel()
    nn_model.train(epochs=25, verbose=0)
    agent.register_module("neural_network", nn_model)

    # 5. Fuzzy Logic Controller (Optional)
    if FuzzySeverityAssessor:
        fuzzy = FuzzySeverityAssessor()
        agent.register_module("fuzzy_controller", fuzzy)

    # 6. STRIPS Treatment Planner
    planner = TreatmentPlanner()
    agent.register_module("planner", planner)

    print(f"\n{C.GREEN}✅ AI Healthcare System Assembly Complete!{C.END}")
    return agent


def create_sample_patients():
    """Generate test patient profiles with diverse symptoms and vitals."""
    return [
        PatientPercept(
            patient_id="P-001",
            symptoms=["Fever", "Cough", "Loss of Smell", "Fatigue"],
            age=34,
            temperature=38.8,
            heart_rate=95,
            blood_pressure="120/80"
        ),
        PatientPercept(
            patient_id="P-002",
            symptoms=["Chest Pain", "Shortness of Breath", "Sweating"],
            age=58,
            temperature=37.2,
            heart_rate=125,
            blood_pressure="150/95"
        ),
        PatientPercept(
            patient_id="P-003",
            symptoms=["Headache", "Stiff Neck", "Fever", "Light Sensitivity"],
            age=22,
            temperature=39.8,
            heart_rate=118,
            blood_pressure="130/85"
        ),
        PatientPercept(
            patient_id="P-004",
            symptoms=["Frequent Urination", "Excessive Thirst", "Fatigue"],
            age=45,
            temperature=36.8,
            heart_rate=78,
            blood_pressure="128/82"
        )
    ]


def render_dashboard(agent_reports: list):
    """Generate summary visualization dashboard."""
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # 1. Diagnoses Summary
    ax1 = fig.add_subplot(gs[0, 0])
    diagnoses = [r['diagnosis'].upper() for r in agent_reports]
    unique_dx, counts = np.unique(diagnoses, return_counts=True)
    ax1.bar(unique_dx, counts, color='#3498db')
    ax1.set_title("Diagnoses Distribution", fontweight='bold')
    ax1.set_ylabel("Patient Count")

    # 2. Confidence Scores
    ax2 = fig.add_subplot(gs[0, 1])
    p_ids = [r['patient_id'] for r in agent_reports]
    confs = [r['confidence'] * 100 for r in agent_reports]
    ax2.barh(p_ids, confs, color='#2ecc71')
    ax2.set_xlim(0, 100)
    ax2.set_title("Agent Diagnostic Confidence (%)", fontweight='bold')
    ax2.set_xlabel("Confidence %")

    # 3. Urgency Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    urgencies = [r['urgency'] for r in agent_reports]
    u_labels, u_counts = np.unique(urgencies, return_counts=True)
    colors = {'CRITICAL': '#e74c3c', 'HIGH': '#e67e22', 'MEDIUM': '#f1c40f', 'LOW': '#2ecc71'}
    bar_colors = [colors.get(u, '#3498db') for u in u_labels]
    ax3.bar(u_labels, u_counts, color=bar_colors)
    ax3.set_title("Urgency Triage Breakdown", fontweight='bold')

    # 4. Treatment Plan Step Counts
    ax4 = fig.add_subplot(gs[1, 1])
    plan_lengths = [
        r['module_results'].get('planner', {}).get('steps', 0)
        for r in agent_reports
    ]
    ax4.plot(p_ids, plan_lengths, marker='o', color='#9b59b6', linewidth=2)
    ax4.set_title("Treatment Plan Steps Generated", fontweight='bold')
    ax4.set_ylabel("Steps Count")

    plt.suptitle("AI Agent Diagnostic System Performance", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig("dashboard.png", dpi=150)
    print(f"\n{C.GREEN}📊 Performance dashboard saved to dashboard.png{C.END}")


def main():
    banner()
    
    # Build System & Load Modules
    agent = build_system()
    patients = create_sample_patients()
    reports = []

    section("🩺 Processing Test Patients")

    for idx, patient in enumerate(patients, 1):
        print(f"\n{C.BOLD}{C.BLUE}--- Patient Case {idx}: {patient.patient_id} ---{C.END}")
        print(f"  Vitals   : Temp={patient.temperature}°C | HR={patient.heart_rate} BPM | BP={patient.blood_pressure}")
        print(f"  Symptoms : {', '.join(patient.symptoms)}")

        # Execute full agent loop: Perceive -> Think -> Act
        report = agent.run(patient)
        reports.append(report)

        # Output Summary
        u_color = C.RED if report['urgency'] in ['CRITICAL', 'HIGH'] else C.GREEN
        print(f"\n  {C.BOLD}Top Diagnosis{C.END} : {report['diagnosis'].upper()}")
        print(f"  {C.BOLD}Confidence{C.END}    : {report['confidence']:.1%}")
        print(f"  {C.BOLD}Triage Urgency{C.END}: {u_color}{report['urgency']}{C.END}")
        
        # Display Generated Treatment Plan
        plan_data = report['module_results'].get('planner', {})
        if 'plan' in plan_data and plan_data['plan']:
            print(f"\n  {C.BOLD}Treatment Plan ({plan_data['steps']} steps):{C.END}")
            for step in plan_data['plan']:
                print(f"    Step {step['step']}: {step['action']} ({step['duration']})")

    # Print internal agent memory/logs
    section("📋 Agent Internal Execution Logs")
    agent.print_log()

    # System Metrics & Dashboards
    section("📈 Agent Performance Dashboard")
    perf = agent.get_performance()
    print(f"  Total Patients Evaluated : {perf['total_patients']}")
    print(f"  Diagnoses Completed     : {perf['diagnoses_made']}")
    print(f"  Cumulative Score         : {perf['performance_score']}")

    render_dashboard(reports)


if __name__ == "__main__":
    main()