DEDAN KIMATHI UNIVERSITY OF SCIENCE AND TECHNOLOGY
AI Capstone Project LAB Manual
CCS 3101 Introduction to Artificial Intelligence — Complete lab Guide
Capstone Overview
Detail Info
Project Title Intelligent AI System: From Agents to Applications
Duration Spans Weeks 8–13 (developed alongside coursework)
Team Size 3–4 students per team
Deliverables Proposal → Midpoint Report → Final System + Presentation
Total Weight 25% of final grade
Capstone Roadmap
Week 8 ──► Topic Selection + Proposal Submission
Week 9 ──► Data Collection + Preprocessing
Week 10 ──► Core AI System Development (Phase 1)
Week 11 ──► Integration + NLP/Application Layer (Phase 2)
Week 12 ──► Evaluation + Optimization (Phase 3)
Week 13 ──► Final Presentation + Demo + Report
Capstone Project: Intelligent Healthcare Diagnostic Assistant
Mission: Build an end-to-end AI system that integrates intelligent agents, search, probabilistic
reasoning, machine learning, NLP, fuzzy logic, and planning into a unified healthcare diagnostic
and recommendation platform.
Step 1 — Prerequisites (Install These First)
Before cloning the project, make sure the following tools are installed on your computer:
Tool Purpose Download Link
Python 3.9+ Programming language python.org
Git Clone & manage code git-scm.com
VS Code or Jupyter Code editor / notebook code.visualstudio.com
pip Python package manager Comes with Python
Step 2 — Clone the Repository
Open your terminal (Mac/Linux) or Command Prompt / PowerShell (Windows) and run the
following commands line by line:
1. Clone the project from GitHub
git clone https://github.com/sirrom/Capstone-Project-Intelligent-Healthcare-Diagnostic-Assistant.git
NB: Code has some incomplete parts that you work on
2. Move into the project folder
cd ai-capstone-healthcare
3. Check the folder contents — you should see all module files
ls # Mac/Linux dir # Windows
Step 3 — Create a Virtual Environment
A virtual environment keeps your project's packages separate from your system Python. This prevents
version conflicts.
# Create the virtual environment (do this ONCE)
python -m venv venv
# Activate it — choose based on your OS:
# Windows:
venv\Scripts\activate
# Mac / Linux:
source venv/bin/activate
# You should now see (venv) at the start of your terminal line 􈆆􎀱
Environment set up
Step 4 — Install All Required Packages
# Install all dependencies
pip install -r requirements.txt
Project folder structure
Student Rule: Never edit files inside modules/ until you have read the module description below.
Understand first, then code.
Step 7 — Working with Git as a Team
Each team member should follow this workflow every session:
# Before starting work — always pull latest changes
git pull origin main
# After finishing your work — save and upload
git add .
git commit -m "Added fuzzy logic module - [Your Name]"
git push origin main
# If your team uses branches (recommended):
git checkout -b feature/your-name-module
# ... do your work ...
git push origin feature/your-name-module
# Then open a Pull Request on GitHub
Team Tip: Assign one module per person to avoid conflicts. Use GitHub Issues to track who is doing
what.
Big Picture — How All Modules Connect
Before diving into individual modules, understand how they fit together as one system:
Key Insight: No single module gives the final answer alone. Each module gives its own diagnosis
and confidence score. The Agent combines them all to produce the best possible recommendation —
just like a real medical team.
Intelligent Agent Core
modules/agent.py
What This Module Does
This is the brain and coordinator of the entire system. It is the first module you will encounter and
the one that ties everything else together.
Think of it like a hospital triage nurse who:
1. Receives a patient (perceives the environment)
2. Consults specialists — the other AI modules (thinks and reasons)
3. Decides what action to take based on all the advice (acts)
In AI terminology, this implements a Model-Based, Goal-Based Agent — the most sophisticated
type of agent that:
 Maintains an internal model of the world (patient history, previous diagnoses)
 Has goals (correctly diagnose and help the patient)
 Learns from interactions over time
AI Concept Behind It
This module directly applies the PEAS Framework and Agent Architecture from Week4
PEAS Component In Our System
Performance Diagnostic accuracy, patient safety, response time
Environment Patient symptoms, vitals, medical history
Actuators Diagnosis report, treatment plan, urgent alert
Sensors Symptom text input, temperature reading, heart rate
The agent follows the classic Perceive → Think → Act loop:
Patient Data ──► perceive () ──► think() ──► act() ──► Report
How To Build It — Step by Step
Step 1: Understand the Data Structure
The agent receives a PatientPercept object every time a new patient arrives. Think of this as the
patient's intake form:
# This represents one patient's data
patient = PatientPercept(
patient_id = "P001",
symptoms = ["fever", "cough", "fatigue", "loss of smell"],
age = 34,
temperature = 38.9, # Celsius
heart_rate = 98, # BPM
blood_pressure = "120/80"
)
Step 2: Build the perceive() method
This method should:
 Store the patient data in the agent's memory
 Record a timestamp
 Change the agent's internal state to COLLECTING
Step 3: Build the think() method
This is the most important step. The agent calls every registered module's.analyze() method and
collects the results:
# Pseudocode for think()
for each module in registered_modules:
result = module.analyze(patient_data)
store result in diagnosis_results
Step 4: Build the act() method
This method:
 Averages the confidence scores from all modules
 Determines urgency (CRITICAL / HIGH / MEDIUM / LOW) based on vitals
 Generates a list of recommendations
 Returns a final structured report
Step 5: Register all modules
In app.py, after creating the agent, you attach each AI sub-module:
agent.register_module('KnowledgeBase', MedicalKnowledgeBase())
agent.register_module('BayesianNet', SimpleBayesianDiagnostics())
... and so on
Expected Output:
[collecting_symptoms] Perceived patient P001 with 2 symptoms Agent test passed!
Common Student Mistakes
Mistake Fix
Calling think() before perceive () Always call perceive () first
Forgetting to register modules Check app.py — all modules must be registered
Confusing Patient Percept with a
dictionary
It is a dataclass — use dot
notation: percept.temperature
MODULE 2 — Medical Knowledge Base & Inference Engine
modules/knowledge_base.py
What This Module Does
This module is the rule-book doctor of the system. It stores medical knowledge as logical rules and
uses inference to draw conclusions from patient symptoms.
Imagine a senior doctor who has memorized thousands of medical rules like:
 "IF a patient has fever AND cough AND loss of smell THEN COVID-19 is suspected"
 "IF COVID-19 is suspected AND fatigue is present THEN COVID-19 is likely"
The module does this automatically using two reasoning strategies:
 Forward Chaining — Start from symptoms, fire rules, reach a diagnosis
 Backward Chaining — Start from a suspected diagnosis, work backwards to see if
symptoms support it
AI Concept Behind It
This module applies First-Order Logic (FOL) and Inference Engines from Week 5.
Concept Where It Appears
Facts Observed symptoms: "fever", "cough"
Rules Medical knowledge: IF fever AND cough THEN flu_possible
Forward Chaining Symptoms → Diagnosis (data-driven)
Backward Chaining Diagnosis → Check Symptoms (goal-driven)
Certainty Factors Each rule has a confidence value (0.0 to 1.0)
Example of a Rule in Our System:
Rule: fever ∧ cough ∧ loss_of_smell ∧ fatigue → covid19_suspected (CF = 0.85)
Meaning: If the patient has ALL FOUR symptoms, then there is an 85% confidence that COVID-19 is
suspected.
How To Build It — Step by Step
Step 1: Understand the knowledge representation
Rules are stored as tuples: (list_of_conditions, conclusion, certainty_factor)
# Example rule entry
(["fever", "cough", "fatigue"], # Conditions
"flu_suspected", # Conclusion
0.75) # Certainty (75% confidence)
Step 2: Implement add_fact() and add_rule()
These are simple methods that store data in the KB's internal lists. Make sure each fact also stores a
certainty factor.
Step 3: Implement load_patient_symptoms()
This translates the patient's symptom list into KB facts. Note the text cleaning:
 "Loss of Smell" → "loss_of_smell" (lowercase + underscores)
Step 4: Implement Forward Chaining
The algorithm works in loops:
Loop:
For each rule:
If ALL conditions are known facts:
Add conclusion as new fact
Record the certainty factor
Repeat until no new facts are added
Stop.
Step 5: Implement Backward Chaining
The algorithm works recursively:
To prove GOAL:
If GOAL is already a known fact → return True
For each rule whose conclusion = GOAL:
Try to prove ALL conditions of that rule
If successful → GOAL is proved
If no rule works → return False
Step 6: Implement analyze()
This is the standard interface method the Agent calls. It should:
1. Clear old facts
2. Load the patient's symptoms as new facts
3. Add vitals (fever = temperature > 38°C)
4. Run forward chaining
5. Return the top diagnosis and confidence
How To Test This Module
from modules. knowledge_base import MedicalKnowledgeBase
kb = MedicalKnowledgeBase()
kb.add_fact("fever")
kb.add_fact("cough")
kb.add_fact("loss_of_smell")
kb.add_fact("fatigue")
results = kb.forward_chain(verbose=True)
print("Inferred:", results)
# Test backward chaining
proved, cf = kb.backward_chain("covid19_suspected")
print (f"COVID-19 suspected: {proved}, Confidence: {cf}")
Expected Output:
Iter 1: fever ∧ cough ∧ loss_of_smell ∧ fatigue → covid19_suspected (CF=0.850)
Inferred: {'covid19_suspected': 0.85, ...}
COVID-19 suspected: True, Confidence: 0.85
Common Student Mistakes
Mistake Fix
Symptoms with spaces not matching
rules
Always convert to "lower_snake_case"
Infinite loop in forward chaining Add a visited set — never re-infer known facts
Backward chaining returns wrong
confidence
Combine CFs by multiplying rule CF × min
(condition CFs)
MODULE 3: Bayesian Network & Probabilistic Reasoning
modules/bayesian_net.py
What This Module Does
This module is the statistical probability calculator of the system. Instead of using hard rules like
Module 2, it reasons under uncertainty using probability theory.
Think of it as a doctor who thinks in percentages:
 "Given these symptoms, there is a 73% chance of flu, 15% chance of COVID, 8% chance of
common cold..."
It uses Bayes' Theorem to update beliefs as new evidence (symptoms) arrives. The more diagnostic
symptoms a patient shows, the more the probability of the matching disease increases.
AI Concept Behind It
This module applies Bayesian Networks and Probability Theory from Weeks 6 and 7.
The core formula used is Naïve Bayes
P(Disease | Symptoms) ∝ P(Disease) × P(Symptom₁|Disease) × P(Symptom₂|Disease) × ...
Term Meaning Example
P(Disease) Prior — how common is this disease? P(flu) = 15% in
population
P(Symptom|Disease) Likelihood — how likely is this symptom
given the disease?
P(fever|flu) = 90%
P(Disease|Symptoms) Posterior — our updated belief after seeing
symptoms
P(flu|fever,cough) = ?
Working in log space: Because we multiply many small probabilities together (which can cause
underflow errors), we use log probabilities and then convert back:
log_prob = log(prior) + log(P(s1|D)) + log(P(s2|D)) + ...
How To Build It — Step by Step
Step 1: Define prior probabilities
These represent the base rate of each disease in the general population:
self. Priors = {
'flu': 0.15, # 15% of sick patients have flu
'covid19': 0.08,
'common_cold': 0.30,
# ...
}
# Note: All priors should add up to ~1.0
Step 2: Define likelihood table
For every disease, estimate the probability of each symptom occurring:
self.likelihoods = {
'flu': {
'fever': 0.90, # 90% of flu patients have fever
'cough': 0.85,
'rash': 0.05, # Only 5% of flu patients have a rash
# ...
}
}
Step 3: Implement compute_posterior()
for each disease:
log_score = log(prior[disease])
for each symptom in patient_symptoms:
p = likelihoods[disease].get(symptom, 0.01) # 0.01 = rare but possible
log_score += log(p)
posteriors[disease] = log_score
# Convert log scores to actual probabilities (softmax-style)
# Divide each by the total so they sum to 1
Step 4: Implement analyze()
Returns the disease with the highest posterior probability along with a ranked list of all diseases.
How To Test This Module
from modules.bayesian_net import SimpleBayesianDiagnostics
bn = SimpleBayesianDiagnostics()
posteriors = bn.compute_posterior(["fever", "cough", "loss_of_smell", "fatigue"])
print("Top 3 Diagnoses:")
ranked = sorted(posteriors.items(), key=lambda x: x[1], reverse=True)
for disease, prob in ranked[:3]:
print(f" {disease:<20}: {prob:.2%}")
Expected Output:
Top 3 Diagnoses:
covid19 : 54.32%
flu : 28.15%
common_cold : 9.87%
Common Student Mistakes
Mistake Fix
Probability of 0 for a
symptom
Use a small floor value like 0.01 instead of 0 — log(0) =
undefined
Forgetting to normalize Divide all posteriors by their sum so they add to 1.0
Symptom names not
matching
Clean symptom strings: "Loss of Smell" → "loss_of_smell"
MODULE 4 — Machine Learning Diagnostic Classifier
modules/ml_classifier.py
What This Module Does
This module is the data-driven pattern recognizer. Unlike Module 2 (which uses human-written
rules) and Module 3 (which uses probability tables), this module learns patterns directly from data.
It trains three different ML algorithms on a synthetic dataset of 2,000 patients and their diagnoses,
then selects the best-performing model automatically.
Think of it as a doctor who studied 2,000 past patient records and learned to recognize patterns —
without being explicitly told the rules.
AI Concept Behind It
This module applies Supervised Learning and Decision Trees .
Algorithm How It Works Strength
Decision Tree Splits data on most informative symptoms using
information gain
Interpretable, visual
Random Forest Builds 100 decision trees and takes the majority vote Robust, handles
noise
Gradient
Boosting
Builds trees sequentially, each correcting previous
errors
Highest accuracy
Feature Vector: Each patient is represented as an 18-dimensional binary vector:
Patient: [fever=1, cough=1, fatigue=1, headache=0, rash=0,
loss_of_smell=1, chest_pain=0, ... (18 symptoms total)]
Label: "covid19"
Evaluation: We use 5-fold Cross Validation to ensure the model generalizes well, not just
memorizing the training data.
How To Build It — Step by Step
Step 1: Understand the symptom feature list
There are 18 binary features (1 = symptom present, 0 = absent). This is the fixed format every patient
record must follow.
Step 2: Generate synthetic training data using _generate_synthetic_data()
Since we do not have a real hospital database, we generate realistic fake data based on known medical
symptom-disease relationships:
# For each disease, define symptom probabilities
# Then randomly generate 250 patients per disease
profiles = {
'flu': {'fever': 0.90, 'cough': 0.85, 'fatigue': 0.88, ...},
'dengue': {'fever': 0.98, 'rash': 0.75, 'joint_pain': 0.85, ...},
# ...
}
Step 3: Train all three models in train()
for each model (Decision Tree, Random Forest, Gradient Boosting):
model.fit(X_train, y_train)
score = cross_val_score(model, X, y, cv=5)
if score > best_score:
select this as best_model
Step 4: Implement predict ()
Convert a list of symptom strings into a binary vector, then call the best
model's .predict() and .predict_proba() methods.
Step 5: Implement plot_evaluation()
This should generate:
 A confusion matrix heatmap
 A feature importance bar chart (which symptoms matter most?)
How To Test This Module
from modules.ml_classifier import MLDiagnosticClassifier
clf = MLDiagnosticClassifier()
clf.train(verbose=True)
result = clf.predict(["fever", "cough", "fatigue", "loss_of_smell"])
print(f"Diagnosis : {result['diagnosis']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Model Used: {result['model_used']}")
# Generate evaluation plots
clf.plot_evaluation()
Common Student Mistakes
Mistake Fix
Passing symptom strings directly to
the model
You MUST convert to a binary vector first using the
feature list
Training on test data Use train_test_split() — never let test data touch the
training process
Overfitting Decision Tree Set max_depth=8 or use pruning
MODULE 5 — Deep Neural Network Diagnostic Model
modules/neural_network.py
What This Module Does
This module is the deep learning specialist of the system. It uses a Multi-Layer Perceptron
(MLP) — a fully connected deep neural network — trained on synthetic patient data to classify
diseases.
While Module 4 uses traditional ML algorithms, this module uses TensorFlow/Keras to build a
modern neural network with:
 Multiple hidden layers
 Batch normalization (stabilizes training)
 Dropout (prevents overfitting)
 Early stopping (stops when performance plateaus)
Think of this as the most advanced specialist in the hospital — slower to train but capable of finding
very complex patterns.
AI Concept Behind It
This module applies Neural Networks and Deep Learning from Week 10.
Network Architecture:
Input Layer → 18 neurons (one per symptom)
Hidden Layer 1 → 128 neurons + BatchNorm + Dropout(30%)
Hidden Layer 2 → 64 neurons + BatchNorm + Dropout(20%)
Hidden Layer 3 → 32 neurons + BatchNorm
Output Layer → 8 neurons (one per disease, Softmax)
Key Techniques:
Technique Purpose
ReLU Activation Introduces non-linearity
Softmax Output Converts scores to probabilities
Dropout Randomly disables neurons during training to prevent memorization
Batch Normalization Normalizes layer outputs for stable training
Early Stopping Halts training when validation accuracy stops improving
L2 Regularization Penalizes large weights to keep the model general
How To Build It — Step by Step
Step 1: Build the model architecture in _build_model()
Use Keras Sequential API:
model = models.Sequential([
layers.Input(shape=(18,)),
layers.Dense(128, activation='relu'),
layers.BatchNormalization(),
layers.Dropout(0.3),
# ... add more layers
layers.Dense(8, activation='softmax') # 8 diseases
])
model.compile(
optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy']
)
Step 2: Generate training data in _generate_data()
Same approach as Module 4 — create synthetic patient records. Aim for 3,000 samples for better
training.
Step 3: Implement train()
Add callbacks for smart training:
callbacks_list = [
EarlyStopping(monitor='val_accuracy', patience=10),
ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
epochs=50, callbacks=callbacks_list)
Step 4: Implement predict ()
Convert symptoms to a binary feature vector, run model.predict(), and return the class with the highest
probability.
Step 5: Implement plot_training()
Plot two graphs side by side:
 Training accuracy vs Validation accuracy over epochs
 Training loss vs Validation loss over epochs
How To Test This Module
from modules.neural_network import NeuralDiagnosticModel
nn = NeuralDiagnosticModel()
nn.train(epochs=30)
result = nn.predict(["fever", "rash", "joint_pain", "headache"])
print(f"Diagnosis : {result['diagnosis']}")
print(f"Confidence: {result['confidence']:.2%}")
nn.plot_training()
Common Student Mistakes
Mistake Fix
Model not converging Make sure features are float32, not integers
Accuracy stuck at 12.5% (1/8
chance)
Your model is guessing randomly — check your data
generation
Training takes forever Reduce epochs to 30 for testing; use GPU if available
MODULE 6 — Fuzzy Logic Severity Assessor
modules/fuzzy_controller.py
What This Module Does
This module answers a different question from the others. Instead of asking "What disease does this
patient have?", it asks: "How serious is this patient's condition right now?"
It produces a severity score from 0 to 100 based on three inputs:
 Temperature (in Celsius)
 Heart Rate (beats per minute)
 Number of symptoms
It uses Fuzzy Logic — a system that handles the grey area between categories. A temperature of
38.1°C is not simply "normal" or "high" — it is somewhat normal and somewhat mild. Fuzzy logic
captures this nuance perfectly.
AI Concept Behind It
This module applies Fuzzy Logic and Fuzzy Control Systems from Week 12.
The Three Steps of Fuzzy Logic:
Step 1: FUZZIFICATION
Convert crisp inputs (e.g., temp = 38.9°C) into
membership degrees across fuzzy sets:
→ normal: 0.1, mild: 0.7, high: 0.3, critical: 0.0
Step 2: RULE EVALUATION
Apply fuzzy IF-THEN rules using min (AND) / max (OR):
→ IF temp IS high AND heart_rate IS elevated
THEN severity IS high (strength = min(0.3, 0.6) = 0.3)
Step 3: DEFUZZIFICATION
Convert fuzzy output back to a single crisp number
using the centroid method:
→ severity_score = 72.4 out of 100
Membership Functions for Temperature:
normal: ████ (peaks at 36.5°C–37.0°C)
mild: ████ (peaks at 38.0°C)
high: ████ (peaks at 39.0°C)
critical: ████ (rises above 39.5°C)
How To Build It — Step by Step
Step 1: Define membership functions
Each input variable needs triangular or trapezoidal membership functions.
Implement _membership_temp(), _membership_hr(), and _membership_symptoms(). Each returns a
dictionary of fuzzy set names and their membership degrees (0.0 to 1.0).
def _membership_temp(self, temp):
return {
'normal': ..., # Calculate degree using triangular formula
'mild': ...,
'high': ...,
'critical': ...
}
Step 2: Define fuzzy rules
Rules combine input memberships using min() for AND and max() for OR: rules = {
'critical': max(
min(temp_mf['critical'], hr_mf['high']), # Rule 1
min(temp_mf['critical'], symptom_mf['many']) # Rule 2
),
'high': max(
min(temp_mf['high'], hr_mf['elevated']),
...
),
# ... more rules for moderate, mild, low
}
Step 3: Implement defuzzification
Use the centroid method:
centers = {'low': 15, 'mild': 35, 'moderate': 55, 'high': 75, 'critical': 92}
score = sum(centers[k] * rules[k] for k in rules) / sum(rules.values())
Step 4: Map score to label
0–20 → LOW
20–40 → MILD
40–60 → MODERATE
60–80 → HIGH
80–100 → CRITICAL
HOW TO TEST
from modules.fuzzy_controller import FuzzySeverityAssessor
fa = FuzzySeverityAssessor()
test_cases = [
(37.0, 72, 2, "Normal patient"),
(38.5, 95, 4, "Mild illness"),
(39.8, 115, 7, "Severe case"),
(40.2, 130, 9, "Critical case"),
]
for temp, hr, count, desc in test_cases:
result = fa.assess(temp, hr, count)
print(f"{desc}: Score={result['severity_score']}, "
f"Label={result['severity_label']}")
Expected Output:
Normal patient : Score=12.3, Label=LOW
Mild illness : Score=38.7, Label=MILD
Severe case : Score=71.2, Label=HIGH
Critical case : Score=89.4, Label=CRITICAL
Common Student Mistakes
Mistake Fix
Membership values exceeding 1.0 Always clamp output with max (0, min (1, value))
Division by zero in defuzzification Add a small epsilon: sum (rules. Values ()) + 1e-10
All rules firing at 0 Your membership functions are not covering all input ranges
MODULE 7 — AI Treatment Planner
modules/planner.py
What This Module Does
This is the decision and action planner of the system. Once the agent knows the diagnosis and
severity, this module generates a step-by-step treatment plan — a sequence of medical actions from
initial state to goal state.
This is exactly like a navigator giving you turn-by-turn directions:
 Where you are now: Patient presents with suspected COVID-19
 Where you want to be Patient is treated, monitored, and has a follow-up scheduled
 The plan: A sequence of specific actions to get from start to goal
AI Concept Behind It
This module applies STRIPS Planning and AI Planning from Week 12, building on Search
Algorithms from Week 3.
STRIPS Representation:
Every action in the medical world is defined as: Action: OrderPCRTest
Preconditions: {COVID_SUSPECTED, PATIENT_PRESENT} ← Must be true BEFORE
Delete List : {COVID_SUSPECTED} ← Becomes false AFTER
Add List : {PCR_PENDING} ← Becomes true AFTER
Planning Algorithm — BFS over State Space:
State 0: {PATIENT_PRESENT, COVID_SUSPECTED, CONTAGIOUS_DISEASE}
↓ Apply: IsolatePatient
State 1: {PATIENT_PRESENT, PATIENT_ISOLATED, DIAGNOSIS_NEEDED}
↓ Apply: OrderPCRTest
State 2: {PATIENT_PRESENT, PATIENT_ISOLATED, PCR_PENDING}
↓ Apply: ReceivePCRResult
State 3: {PATIENT_PRESENT, PATIENT_ISOLATED, DIAGNOSIS_CONFIRMED}
↓ ...
GOAL : {TREATMENT_STARTED, VITALS_MONITORED, FOLLOWUP_SCHEDULED}
How To Build It — Step by Step
Step 1: Define the action library
Create a list of medical actions. Each action is a dictionary with:
name — readable name
precond — set of facts that must be true
delete — set of facts removed after the action
add — set of facts added after the action
cost and duration — metadata
Step 2: Implement _apply_action()
def _apply_action(self, state, action):
if not action['precond'].issubset(state):
return None # Cannot apply this action
new_state = (state - action['delete']) | action['add']
return frozenset(new_state)
Step 3: Implement generate_plan() using BFS
Use a queue to explore possible action sequences:
queue = deque([(initial_state, [])]) # (current_state, plan_so_far)
while queue:
state, plan = queue.popleft()
if goal.issubset(state):
return plan # Found a valid plan!
for action in all_actions:
new_state = apply_action(state, action)
if new_state and new_state not in visited:
queue.append((new_state, plan + [action]))
Step 4: Implement create_treatment_plan()
This maps a diagnosis name to the appropriate initial state predicates and calls generate_plan()
How To Test This Module
from modules.planner import TreatmentPlanner
planner = TreatmentPlanner()
# Test for COVID-19 case
plan = planner.create_treatment_plan('covid19', 'HIGH')
print(f"Diagnosis : {plan['diagnosis']}")
print(f"Plan Steps: {plan['steps']}")
print()
for step in plan['plan']:
print(f" Step {step['step']:2d}: {step['action']:<30} [{step['duration']}]")
Expected Output:
Diagnosis : covid19
Plan Steps: 6
Step 1: IsolatePatient [14 days]
Step 2: OrderPCRTest [24 hours]
Step 3: ReceivePCRResult [24 hours]
Step 4: PrescribeAntiviral [10 minutes]
Step 5: MonitorVitals [Continuous]
Step 6: ScheduleFollowUp [5 minutes]
Common Student Mistakes
Mistake Fix
BFS never finding a plan Check that goal_state predicates are actually reachable through
your actions
Plan is unrealistically long Limit search depth or add more complete actions
Action loops (same state
revisited)
Always maintain a visited set of seen states
Evaluation Module
evaluation/metrics.py and evaluation/visualizations.py
What This Module Does
This module measures how well the entire AI system performs. Think of it as the quality assurance
department of the hospital that audits all diagnoses and reports accuracy statistics.
It compares the diagnoses from all modules against known ground-truth labels and generates
comprehensive performance reports.
Metrics You Must Report
Metric Formula What It Tells You
Accuracy Correct / Total Overall correctness
Precision TP / (TP + FP) How often positive predictions are right
Recall TP / (TP + FN) How often actual positives are caught
F1-Score 2 × (P × R) / (P + R) Balance of precision and recall
Confusion Matrix Grid of predictions vs actual Which diseases get confused
ROC-AUC Area under ROC curve Discrimination ability
Final deliverables
[ ] All 7 modules complete and integrated
[ ] app.py runs end-to-end without errors
[ ] Evaluation metrics computed for all ML modules
[ ] Confusion matrices generated for each classifier
[ ] Module comparison bar chart generated
[ ] At least 5 test patients diagnosed by the full system
[ ] Final report PDF in reports/ folder
[ ] 10-minute live demo prepared
[ ] README.md updated with setup instructions and results
If you are stuck, follow this order:
1. Re-read this guide for the specific module
2. Check the module's test code — does the expected output match?
3. Read the error message carefully — Python errors are descriptive
4. Search: "sklearn RandomForest example" or "Keras Dense layer tutorial"
5. Ask your teammate — pair programming helps
6. Post in the class discussion forum with:
- Which module you are working on
- What you expected to happen
- What actually happened (paste the error)
7. Contact your instructor during office hours
Pro Tip: Run each module independently first using its test code. Only integrate into app.py after
each module works on its own.
