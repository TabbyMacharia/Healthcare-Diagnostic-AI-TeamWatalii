from collections import deque
from typing import Dict, List, Set, Optional

class TreatmentPlanner:
    """
    STRIPS-based treatment planner.
    Generates step-by-step treatment plans
    from patient diagnosis to recovery.
    """

def __init__(self):
    self.action_library = self._build_action_library()

    self.diagnosis_states = {
        'flu': {'VIRAL_INFECTION', 'DIAGNOSIS_NEEDED'},
        'covid19': {
            'COVID_SUSPECTED',
            'CONTAGIOUS_DISEASE',
            'DIAGNOSIS_NEEDED'
        },
        'cardiac_event': {
            'EMERGENCY_CASE',
            'ICU_AVAILABLE'
        },
        'dengue': {
            'VIRAL_INFECTION',
            'DIAGNOSIS_NEEDED',
            'DEHYDRATION_RISK'
        },
        'meningitis': {
            'EMERGENCY_CASE',
            'BACTERIAL_INFECTION',
            'ICU_AVAILABLE'
        },
        'tuberculosis': {
            'BACTERIAL_INFECTION',
            'CONTAGIOUS_DISEASE',
            'DIAGNOSIS_NEEDED'
        },
        'diabetes': {
            'DIAGNOSIS_NEEDED'
        },
        'common_cold': {
            'VIRAL_INFECTION',
            'DIAGNOSIS_NEEDED'
        }
    }

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

    def create_treatment_plan(self, diagnosis: str,
                              urgency: str) -> Dict:
        """Generate a treatment plan for a given diagnosis"""

        # Map diagnosis to initial state predicates
        diagnosis_states = {
            'flu':          {'VIRAL_INFECTION', 'DIAGNOSIS_NEEDED'},
            'covid19':      {'COVID_SUSPECTED', 'CONTAGIOUS_DISEASE',
                             'DIAGNOSIS_NEEDED'},
            'cardiac_event':{'EMERGENCY_CASE',  'ICU_AVAILABLE'},
            'dengue':       {'VIRAL_INFECTION',  'DIAGNOSIS_NEEDED',
                             'DEHYDRATION_RISK'},
            'meningitis':   {'EMERGENCY_CASE',  'BACTERIAL_INFECTION',
                             'ICU_AVAILABLE'},
            'tuberculosis': {'BACTERIAL_INFECTION', 'CONTAGIOUS_DISEASE',
                             'DIAGNOSIS_NEEDED'},
            'diabetes':     {'DIAGNOSIS_NEEDED'},
            'common_cold':  {'VIRAL_INFECTION', 'DIAGNOSIS_NEEDED'},
        }

        base_state = {'PATIENT_PRESENT'}
        dx_state   = diagnosis_states.get(
            diagnosis.lower().replace(' ', '_'),
            {'DIAGNOSIS_NEEDED'}
        )
        initial_state = base_state | dx_state

        # Goal state: always end with treatment and monitoring
        goal_state = {'TREATMENT_STARTED', 'VITALS_MONITORED',
                      'FOLLOWUP_SCHEDULED'}
        if urgency == 'CRITICAL':
            goal_state.add('PATIENT_IN_ICU')

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
        """
        Estimate the total duration of a treatment plan.
        Fixed-time actions are summed, while long-running
        activities (e.g. Continuous, 14 days) are listed separately.
        """

        total_minutes = 0
        special = []

        for action in plan:
            duration = action["duration"]

            if duration == "Continuous":
                special.append("Continuous monitoring")

            elif "day" in duration:
                special.append(duration)

            elif "hour" in duration:
                value = int(duration.split()[0])
                total_minutes += value * 60

            elif "minute" in duration:
                value = int(duration.split()[0])
                total_minutes += value

        hours = total_minutes // 60
        minutes = total_minutes % 60

        summary = []

        if hours:
            summary.append(f"{hours} hr")

        if minutes:
            summary.append(f"{minutes} min")

        if special:
            summary.extend(special)

        return ", ".join(summary)

    def analyze(self, percept) -> Dict:
        """
        Module interface.
        Generates a treatment plan using the diagnosis
        and urgency provided by previous AI modules.
        """

        diagnosis = getattr(percept, "diagnosis", "flu")
        urgency = getattr(percept, "urgency", "MEDIUM")
        confidence = getattr(percept, "confidence", 1.0)

        result = self.create_treatment_plan(diagnosis, urgency)

        if "error" not in result:
            result["summary"] = (
                f"Treatment plan generated for {diagnosis} "
                f"({urgency})"
            )
            result["confidence"] = confidence

        return result
# Module completed.