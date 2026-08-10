from copy import deepcopy
from collections import deque
from typing import Dict, List, Set, Tuple, Optional

class TreatmentPlanner:
    """
    STRIPS-based treatment planner.
    Generates step-by-step treatment plans
    from patient diagnosis to recovery.
    """

    def __init__(self):
        self.action_library = self._build_action_library()

    def _build_action_library(self) -> List[Dict]:
        """Define medical treatment actions"""
        return [
            # Emergency Actions
            {
                'name': 'CallEmergencyServices',
                'precond': {'EMERGENCY_CASE', 'PATIENT_PRESENT'},
                'delete':  {'EMERGENCY_CASE'},
                'add':     {'EMERGENCY_SERVICES_CALLED'},
                'cost': 0, 'duration': '5 minutes'
            },
            {
                'name': 'TransferToICU',
                'precond': {'EMERGENCY_SERVICES_CALLED', 'ICU_AVAILABLE'},
                'delete':  {'EMERGENCY_SERVICES_CALLED'},
                'add':     {'PATIENT_IN_ICU', 'MONITORING_ACTIVE'},
                'cost': 0, 'duration': '15 minutes'
            },
            # Diagnostics
            {
                'name': 'OrderBloodPanel',
                'precond': {'PATIENT_PRESENT', 'DIAGNOSIS_NEEDED'},
                'delete':  {'DIAGNOSIS_NEEDED'},
                'add':     {'BLOOD_RESULTS_PENDING'},
                'cost': 1, 'duration': '30 minutes'
            },
            {
                'name': 'ReceiveBloodResults',
                'precond': {'BLOOD_RESULTS_PENDING'},
                'delete':  {'BLOOD_RESULTS_PENDING'},
                'add': {'BLOOD_RESULTS_AVAILABLE','DIAGNOSIS_REFINED','DIAGNOSIS_CONFIRMED'},
                'cost': 0, 'duration': '2 hours'
            },
            {
                'name': 'OrderPCRTest',
                'precond': {'COVID_SUSPECTED', 'PATIENT_PRESENT'},
                'delete':  {'COVID_SUSPECTED'},
                'add':     {'PCR_PENDING'},
                'cost': 1, 'duration': '24 hours'
            },
            {
                'name': 'ReceivePCRResult',
                'precond': {'PCR_PENDING'},
                'delete':  {'PCR_PENDING'},
                'add':     {'PCR_RESULT_AVAILABLE', 'DIAGNOSIS_CONFIRMED'},
                'cost': 0, 'duration': '24 hours'
            },
            # Treatment
            {
                'name': 'PrescribeAntiviral',
                'precond': {'DIAGNOSIS_CONFIRMED', 'VIRAL_INFECTION'},
                'delete':  {'VIRAL_INFECTION'},
                'add':     {'ANTIVIRAL_PRESCRIBED', 'TREATMENT_STARTED'},
                'cost': 1, 'duration': '10 minutes'
            },
            {
                'name': 'PrescribeAntibiotics',
                'precond': {'DIAGNOSIS_CONFIRMED', 'BACTERIAL_INFECTION'},
                'delete':  {'BACTERIAL_INFECTION'},
                'add':     {'ANTIBIOTICS_PRESCRIBED', 'TREATMENT_STARTED'},
                'cost': 1, 'duration': '10 minutes'
            },
            {
                'name': 'AdministerFluids',
                'precond': {'PATIENT_IN_ICU', 'DEHYDRATION_RISK'},
                'delete':  {'DEHYDRATION_RISK'},
                'add':     {'FLUIDS_ADMINISTERED'},
                'cost': 1, 'duration': '1 hour'
            },
            {
                # Ward-level rehydration for non-ICU cases (e.g. Food
                # Poisoning, Typhoid) — AdministerFluids above only fires
                # once a patient is already in the ICU, which left milder
                # dehydration-risk cases with no route to TREATMENT_STARTED.
                'name': 'AdministerOralRehydration',
                'precond': {'DIAGNOSIS_CONFIRMED', 'DEHYDRATION_RISK'},
                'delete':  {'DEHYDRATION_RISK'},
                'add':     {'REHYDRATION_STARTED', 'TREATMENT_STARTED'},
                'cost': 1, 'duration': '1 hour'
            },
            {
                # Malaria is parasitic, not viral/bacterial — needs its
                # own medication action distinct from PrescribeAntiviral/
                # PrescribeAntibiotics.
                'name': 'PrescribeAntimalarial',
                'precond': {'DIAGNOSIS_CONFIRMED', 'PARASITIC_INFECTION'},
                'delete':  {'PARASITIC_INFECTION'},
                'add':     {'ANTIMALARIAL_PRESCRIBED', 'TREATMENT_STARTED'},
                'cost': 1, 'duration': '10 minutes'
            },
            {
                # Chronic conditions (Hypertension) — ongoing management,
                # not a cure, but still counts as "treatment started".
                'name': 'PrescribeAntihypertensive',
                'precond': {'DIAGNOSIS_CONFIRMED', 'CHRONIC_CONDITION'},
                'delete':  {'CHRONIC_CONDITION'},
                'add':     {'MEDICATION_PRESCRIBED', 'TREATMENT_STARTED'},
                'cost': 1, 'duration': '10 minutes'
            },
            {
                # Migraine — symptomatic pain relief.
                'name': 'PrescribePainRelief',
                'precond': {'DIAGNOSIS_CONFIRMED', 'NEUROLOGICAL_CONDITION'},
                'delete':  {'NEUROLOGICAL_CONDITION'},
                'add':     {'PAIN_RELIEF_PRESCRIBED', 'TREATMENT_STARTED'},
                'cost': 1, 'duration': '10 minutes'
            },
            {
                'name': 'MonitorVitals',
                'precond': {'TREATMENT_STARTED', 'PATIENT_PRESENT'},
                'delete':  set(),
                'add':     {'VITALS_MONITORED'},
                'cost': 0, 'duration': 'Continuous'
            },
            {
                'name': 'IsolatePatient',
                'precond': {'CONTAGIOUS_DISEASE', 'PATIENT_PRESENT'},
                'delete':  {'CONTAGIOUS_DISEASE'},
                'add':     {'PATIENT_ISOLATED'},
                'cost': 0, 'duration': '14 days'
            },
            {
                'name': 'ScheduleFollowUp',
                'precond': {'TREATMENT_STARTED', 'VITALS_MONITORED'},
                'delete':  set(),
                'add':     {'FOLLOWUP_SCHEDULED', 'PLAN_COMPLETE'},
                'cost': 0, 'duration': '5 minutes'
            },
            {
                'name': 'DischargePatient',
                'precond': {'PLAN_COMPLETE', 'SYMPTOMS_RESOLVED'},
                'delete':  {'PLAN_COMPLETE'},
                'add':     {'PATIENT_DISCHARGED'},
                'cost': 0, 'duration': '30 minutes'
            },
            {
               'name': 'StabilizeCardiacPatient',
               'precond': {'PATIENT_IN_ICU'},
               'delete': set(),
               'add': {'TREATMENT_STARTED'},
               'cost': 0,
               'duration': '20 minutes'
            },
        ]

    def _apply_action(self, state: frozenset,
                      action: Dict) -> Optional[frozenset]:
        if not action['precond'].issubset(state):
            return None
        return frozenset((state - action['delete']) | action['add'])

    def generate_plan(self,
                      initial_state: Set[str],
                      goal_state:    Set[str]) -> Optional[List[Dict]]:
        """BFS-based plan generation"""
        initial = frozenset(initial_state)
        goal    = frozenset(goal_state)

        queue   = deque([(initial, [])])
        visited = {initial}

        while queue:
            state, plan = queue.popleft()
            if goal.issubset(state):
                return plan

            for action in self.action_library:
                new_state = self._apply_action(state, action)
                if new_state and new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, plan + [action]))

        return None

    # Maps each disease name (as produced by ml_classifier.py /
    # neural_network.py / bayesian_net.py / knowledge_base.py, i.e. the
    # exact Diagnosis values in data/patient_records.csv) to the STRIPS
    # initial-state predicates that get this patient's plan started.
    DIAGNOSIS_STATES = {
        'covid_19':       {'COVID_SUSPECTED', 'CONTAGIOUS_DISEASE',
                           'VIRAL_INFECTION', 'DIAGNOSIS_NEEDED'},
        'influenza':      {'VIRAL_INFECTION', 'DIAGNOSIS_NEEDED'},
        'common_cold':    {'VIRAL_INFECTION', 'DIAGNOSIS_NEEDED'},
        'pneumonia':      {'BACTERIAL_INFECTION', 'DIAGNOSIS_NEEDED'},
        'malaria':        {'PARASITIC_INFECTION', 'DIAGNOSIS_NEEDED'},
        'typhoid':        {'BACTERIAL_INFECTION', 'DIAGNOSIS_NEEDED', 'DEHYDRATION_RISK'},
        'hypertension':   {'CHRONIC_CONDITION', 'DIAGNOSIS_NEEDED'},
        'migraine':       {'NEUROLOGICAL_CONDITION', 'DIAGNOSIS_NEEDED'},
        'food_poisoning': {'DEHYDRATION_RISK', 'DIAGNOSIS_NEEDED'},
    }

    def create_treatment_plan(self, diagnosis: str,
                              urgency: str) -> Dict:
        """Generate a treatment plan for a given diagnosis"""

        norm = diagnosis.lower().replace(' ', '_').replace('-', '_')

        # A "Healthy" diagnosis needs no treatment plan at all.
        if norm == 'healthy':
            return {
                'diagnosis': diagnosis, 'urgency': urgency,
                'initial_state': [], 'goal_state': [],
                'steps': 0, 'total_duration': 'N/A', 'plan': [],
                'note': 'No treatment required — patient is healthy.'
            }

        base_state = {'PATIENT_PRESENT'}
        dx_state = self.DIAGNOSIS_STATES.get(norm, {'DIAGNOSIS_NEEDED'})
        initial_state = base_state | dx_state

        # Goal state: always end with treatment and monitoring
        goal_state = {'TREATMENT_STARTED', 'VITALS_MONITORED',
                      'FOLLOWUP_SCHEDULED'}

        # NOTE on a bug fixed here: the original code added PATIENT_IN_ICU
        # to the *goal* for any CRITICAL case, but PATIENT_IN_ICU is only
        # reachable via TransferToICU, which needs EMERGENCY_SERVICES_CALLED
        # + ICU_AVAILABLE in the *initial* state. Only 'cardiac_event' ever
        # set those, so BFS would return "no plan found" for e.g. a CRITICAL
        # Pneumonia case. Fix: put the patient on the emergency track in the
        # initial state instead, and let the existing action chain
        # (CallEmergencyServices -> TransferToICU -> StabilizeCardiacPatient)
        # carry them to TREATMENT_STARTED regardless of the specific disease.
        if urgency == 'CRITICAL':
            initial_state |= {'EMERGENCY_CASE', 'ICU_AVAILABLE'}

        plan = self.generate_plan(initial_state, goal_state)

        if plan is None:
            return {'error': 'No plan found', 'plan': []}

        return {
            'diagnosis':     diagnosis,
            'urgency':       urgency,
            'initial_state': sorted(initial_state),
            'goal_state':    sorted(goal_state),
            'steps':         len(plan),
            'total_duration': self._estimate_duration(plan),
            'plan': [
                {
                    'step':     i+1,
                    'action':   a['name'],
                    'duration': a['duration'],
                    'cost':     a['cost']
                }
                for i, a in enumerate(plan)
            ]
        }

    def _estimate_duration(self, plan: List[Dict]) -> str:
        durations = [a['duration'] for a in plan]
        return f"{len(plan)} actions | see individual durations"

    def analyze(self, percept) -> Dict:
        """
        Module interface for the agent.

        IMPORTANT: agent.think() calls .analyze() on every registered
        module and folds any 'diagnosis'/'confidence' keys it finds into
        the cross-module vote (see agent.py's _aggregate_diagnosis and
        act()). The planner doesn't diagnose anything — the real
        treatment plan is generated afterwards in app.py's
        show_treatment_plan(), once the Agent already knows the actual
        diagnosis. So this deliberately returns no 'diagnosis' or
        'confidence' key: including a guessed one here would silently
        pollute the vote from the 4 real diagnostic modules on every run.
        """
        return {
            'summary': "Treatment planning deferred until diagnosis is confirmed"
        }
# Module completed.