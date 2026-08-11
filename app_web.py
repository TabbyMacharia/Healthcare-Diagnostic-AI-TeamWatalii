import warnings
warnings.filterwarnings("ignore")

import streamlit as st

# ============================================================
# Page Setup
# ============================================================
st.set_page_config(
    page_title="Healthcare Diagnostic AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Load AI Modules & Initialize Agent
# ============================================================
@st.cache_resource
def load_healthcare_system():
    """Initializes and caches the agent and modules so they only load once."""
    from modules.agent import HealthcareDiagnosticAgent
    from modules.knowledge_base import MedicalKnowledgeBase
    from modules.bayesian_net import SimpleBayesianDiagnostics
    from modules.ml_classifier import MLDiagnosticClassifier
    from modules.neural_network import NeuralDiagnosticModel
    from modules.fuzzy_controller import FuzzySeverityAssessor
    from modules.planner import TreatmentPlanner

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
        agent.register_module(name, module)

    return agent

# Import PatientPercept
from modules.agent import PatientPercept

# Initialize Agent
agent = load_healthcare_system()

# Initialize Session State
if "diagnostic_history" not in st.session_state:
    st.session_state.diagnostic_history = []

# ============================================================
# Header Banner
# ============================================================
st.title("🏥 INTELLIGENT HEALTHCARE DIAGNOSTIC AI SYSTEM")
st.caption("Artificial Intelligence Capstone Project")
st.divider()

# ============================================================
# Sidebar: Patient Input Form
# ============================================================
with st.sidebar:
    st.header("📋 PATIENT INFORMATION")
    
    with st.form("patient_form"):
        patient_id = st.text_input("Patient ID", value="345")
        age = st.number_input("Age", min_value=0, max_value=120, value=56)
        temperature = st.number_input("Temperature (°C)", min_value=30.0, max_value=67.0, value=37.5, step=0.1)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=220, value=97)
        blood_pressure = st.text_input("Blood Pressure (e.g. 120/80)", value="88")
        
        symptoms_raw = st.text_area(
            "Symptoms (separated by commas)", 
            value="headache",
            help="Example: fever,cough,fatigue,headache"
        )
        
        submit_btn = st.form_submit_button("Run AI Diagnosis", type="primary", use_container_width=True)

# ============================================================
# Tabs Setup
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["🩺 Final Report", "🧩 Subsystem Results", "📈 Performance", "📜 Agent Log"])

if submit_btn:
    # Process symptoms list matching CLI logic
    symptom_list = [
        s.strip().lower().replace(" ", "_")
        for s in symptoms_raw.split(",")
        if s.strip()
    ]
    
    # Construct Patient Percept
    patient = PatientPercept(
        patient_id=patient_id,
        symptoms=symptom_list,
        age=int(age),
        temperature=float(temperature),
        heart_rate=int(heart_rate),
        blood_pressure=blood_pressure
    )
    
    with st.spinner("Running Intelligent Agent and AI Modules..."):
        agent.perceive(patient)
        module_results = agent.think()
        report = agent.act(module_results)
        
        # Save to state
        st.session_state["latest_report"] = report
        st.session_state["latest_module_results"] = module_results

# Render content if diagnosis exists
if "latest_report" in st.session_state:
    report = st.session_state["latest_report"]
    module_results = st.session_state["latest_module_results"]

    # --------------------------------------------------------
    # TAB 1: FINAL DIAGNOSIS REPORT & TREATMENT PLAN
    # --------------------------------------------------------
    with tab1:
        st.subheader("FINAL DIAGNOSIS REPORT")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Patient ID", report.get("patient_id", "N/A"))
        col2.metric("Diagnosis", report.get("diagnosis", "Unknown"))
        col3.metric("Confidence", f"{report.get('confidence', 0):.2%}")
        
        urgency = report.get("urgency", "Normal")
        if urgency.upper() in ["HIGH", "CRITICAL", "SEVERE"]:
            col4.error(f"Urgency: {urgency}")
        else:
            col4.info(f"Urgency: {urgency}")

        st.write(f"**Next Action:** `{report.get('next_action', 'N/A')}`")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Symptoms:")
            for symptom in report.get("symptoms", []):
                st.write(f"• `{symptom}`")
                
        with c2:
            st.markdown("### Recommendations:")
            for rec in report.get("recommendations", []):
                st.success(f"✓ {rec}")

        st.divider()

        # Treatment Planner Section
        st.subheader("AI TREATMENT PLAN")
        planner = agent._modules.get("TreatmentPlanner")
        
        if planner:
            plan = planner.create_treatment_plan(report["diagnosis"], report["urgency"])
            
            if "error" in plan:
                st.warning(plan["error"])
            elif plan.get("plan"):
                st.write(f"**Diagnosis:** {plan.get('diagnosis')}")
                st.write(f"**Urgency:** {plan.get('urgency')}")
                st.write(f"**Total Duration:** {plan.get('total_duration')}")
                
                st.markdown("#### Treatment Steps")
                for step in plan.get("plan", []):
                    st.write(f"{step['step']}. **{step['action']}** ({step['duration']})")
            else:
                st.info("No plan found")
        else:
            st.info("Treatment planner not available.")

    # --------------------------------------------------------
    # TAB 2: INDIVIDUAL AI MODULE RESULTS (Matches Terminal Output)
    # --------------------------------------------------------
    with tab2:
        st.subheader("INDIVIDUAL AI MODULE RESULTS")
        st.caption("Detailed predictions output by each subsystem")

        grid_cols = st.columns(2)
        idx = 0
        
        for module_name, result in module_results.items():
            col = grid_cols[idx % 2]
            with col:
                with st.container(border=True):
                    st.markdown(f"### [{module_name}]")
                    
                    if isinstance(result, dict):
                        diagnosis = result.get("diagnosis", "Unknown")
                        confidence = float(result.get("confidence", 0.0))
                        summary = result.get("summary", "")
                        
                        st.write(f"**Diagnosis :** `{diagnosis}`")
                        st.write(f"**Confidence:** `{confidence:.2%}`")
                        
                        # Display confidence bar
                        st.progress(min(max(confidence, 0.0), 1.0))
                        
                        if summary:
                            st.write(f"**Summary   :** {summary}")
                    else:
                        st.write(f"Result: {result}")
            idx += 1

    # --------------------------------------------------------
    # TAB 3: AGENT PERFORMANCE
    # --------------------------------------------------------
    with tab3:
        st.subheader("AGENT PERFORMANCE")
        perf = agent.get_performance()
        
        p_col1, p_col2, p_col3 = st.columns(3)
        p_col1.metric("Patients Diagnosed", perf.get("total_patients", 0))
        p_col2.metric("Performance Score", perf.get("performance_score", 0))
        p_col3.metric("Diagnoses Made", perf.get("diagnoses_made", 0))

    # --------------------------------------------------------
    # TAB 4: AGENT LOG
    # --------------------------------------------------------
    with tab4:
        st.subheader("AGENT LOG")
        st.markdown("**Agent Action Log:**")
        
        # Display formatted log output matching terminal
        if hasattr(agent, "log") and isinstance(agent.log, list):
            log_text = "\n".join(agent.log)
            st.code(log_text, language="text")
        elif hasattr(agent, "print_log"):
            # Fallback if agent prints or returns log string
            st.code("Check terminal for agent.print_log() or view stored session states.", language="text")

else:
    with tab1:
        st.info("👈 Enter patient details in the sidebar and click 'Run AI Diagnosis' to get started.")