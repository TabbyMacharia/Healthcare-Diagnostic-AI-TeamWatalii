from typing import Dict
import numpy as np


class FuzzySeverityAssessor:
    """
    Fuzzy logic system for patient severity assessment.

    Inputs:
        Temperature
        Heart Rate
        Symptom Count
        Oxygen Saturation
        Age

    Output:
        Severity Score (0-100)
    """

    def _membership_temp(self, temp: float) -> Dict[str, float]:
        """Temperature membership functions (Celsius)"""
        return {
            'normal': float(np.clip((37.5 - temp) / 1.0, 0, 1))
            if temp <= 37.5 else 0.0,

            'mild': float(max(0.0, 1.0 - abs(temp - 38.0) / 1.0)),

            'high': float(max(0.0, 1.0 - abs(temp - 39.0) / 1.0)),

            'critical': float(np.clip((temp - 39.0) / 1.5, 0, 1))
            if temp >= 39.0 else 0.0
        }


    def _membership_hr(self, hr: int) -> Dict[str, float]:
        """Heart rate membership functions (BPM)"""
        return {
            'low': float(np.clip((70 - hr) / 10.0, 0, 1))
            if hr <= 70 else 0.0,

            'normal': float(max(0.0, 1.0 - abs(hr - 80) / 20.0)),

            'elevated': float(max(0.0, 1.0 - abs(hr - 100) / 15.0)),

            'high': float(np.clip((hr - 100) / 20.0, 0, 1))
            if hr >= 100 else 0.0
        }


    def _membership_symptoms(self, count: int) -> Dict[str, float]:
        """Symptom count membership functions"""
        return {
            'few': float(np.clip((3 - count) / 2.0, 0, 1)),

            'moderate': float(max(0.0, 1.0 - abs(count - 4) / 2.0)),

            'many': float(np.clip((count - 3) / 3.0, 0, 1))
        }


    def _membership_spo2(self, spo2: float) -> Dict[str, float]:
        """Oxygen saturation membership (%)"""
        return {
            'normal':
                float(np.clip((spo2 - 95) / 3, 0, 1)),

            'low':
                float(np.clip((95 - spo2) / 5, 0, 1)),

            'critical':
                float(np.clip((90 - spo2) / 10, 0, 1))
        }


    def _membership_age(self, age: int) -> Dict[str, float]:
        """Age risk membership"""
        return {
            'young':
                float(np.clip((40 - age) / 20, 0, 1)),

            'adult':
                float(max(0.0, 1 - abs(age - 50) / 30)),

            'elderly':
                float(np.clip((age - 60) / 30, 0, 1))
        }



    def _defuzzify(self, severity_rules: Dict[str, float]) -> float:
        """Centroid defuzzification"""

        centers = {
            'low': 15.0,
            'mild': 35.0,
            'moderate': 55.0,
            'high': 75.0,
            'critical': 92.0
        }

        numerator = sum(
            centers[k] * v
            for k, v in severity_rules.items()
        )

        denominator = sum(severity_rules.values()) + 1e-10

        return numerator / denominator



    def assess(
        self,
        temperature: float,
        heart_rate: int,
        symptom_count: int,
        spo2: float,
        age: int
    ) -> Dict:

        """Full fuzzy inference pipeline"""


        # -------------------------
        # Fuzzification
        # -------------------------

        temp_mf = self._membership_temp(
            temperature
        )

        hr_mf = self._membership_hr(
            heart_rate
        )

        symptom_mf = self._membership_symptoms(
            symptom_count
        )

        spo2_mf = self._membership_spo2(
            spo2
        )

        age_mf = self._membership_age(
            age
        )



        # -------------------------
        # Rule Evaluation
        # -------------------------

        rules = {


            'critical': max(

                min(
                    temp_mf['critical'],
                    hr_mf['high']
                ),

                min(
                    spo2_mf['critical'],
                    symptom_mf['many']
                ),

                min(
                    age_mf['elderly'],
                    spo2_mf['low']
                )
            ),



            'high': max(

                min(
                    temp_mf['high'],
                    hr_mf['elevated']
                ),

                min(
                    temp_mf['high'],
                    symptom_mf['many']
                ),

                min(
                    spo2_mf['low'],
                    symptom_mf['moderate']
                ),

                min(
                    temp_mf['mild'],
                    hr_mf['high']
                )
            ),



            'moderate': max(

                min(
                    temp_mf['mild'],
                    hr_mf['normal']
                ),

                min(
                    temp_mf['high'],
                    symptom_mf['moderate']
                ),

                min(
                    temp_mf['normal'],
                    symptom_mf['many']
                )
            ),



            'mild': max(

                min(
                    temp_mf['mild'],
                    symptom_mf['few']
                ),

                min(
                    temp_mf['normal'],
                    symptom_mf['moderate']
                )
            ),



            'low':

                min(
                    temp_mf['normal'],
                    hr_mf['normal'],
                    spo2_mf['normal'],
                    symptom_mf['few']
                )
        }



        # -------------------------
        # Defuzzification
        # -------------------------

        severity_score = self._defuzzify(
            rules
        )

        severity_label = self._classify(
            severity_score
        )


        return {

            'severity_score':
                round(severity_score,2),

            'severity_label':
                severity_label,

            'rule_strengths':
            {
                k:round(v,3)
                for k,v in rules.items()
            },

            'memberships':
            {
                'temperature': temp_mf,
                'heart_rate': hr_mf,
                'symptoms': symptom_mf,
                'oxygen': spo2_mf,
                'age': age_mf
            }
        }



    def _classify(self, score: float) -> str:

        if score >= 80:
            return "CRITICAL"

        elif score >= 60:
            return "HIGH"

        elif score >= 40:
            return "MODERATE"

        elif score >= 20:
            return "MILD"

        return "LOW"



# ============================================================
# Interactive CLI Execution
# ============================================================

if __name__ == "__main__":

    assessor = FuzzySeverityAssessor()


    print(
        "=== PATIENT SEVERITY ASSESSMENT SYSTEM ==="
    )


    try:

        temp = float(
            input("Enter Patient Temperature (°C): ")
        )


        hr = int(
            input("Enter Heart Rate (BPM): ")
        )


        symptoms = int(
            input("Enter Symptom Count: ")
        )


        spo2 = float(
            input("Enter Oxygen Saturation (%): ")
        )


        age = int(
            input("Enter Patient Age: ")
        )



        result = assessor.assess(

            temperature=temp,

            heart_rate=hr,

            symptom_count=symptoms,

            spo2=spo2,

            age=age

        )



        print("\n" + "="*40)

        print("          ASSESSMENT RESULTS")

        print("="*40)


        print(
            f"Severity Classification : {result['severity_label']}"
        )


        print(
            f"Severity Score          : {result['severity_score']} / 100"
        )



        print("\n--- Rule Strengths ---")

        for level,strength in result['rule_strengths'].items():

            print(
                f"{level.capitalize():<10}: {strength}"
            )



        print("\n--- Membership Values ---")

        for metric,values in result['memberships'].items():

            print(
                metric.capitalize(),
                values
            )


    except ValueError:

        print(
            "\n[Error] Invalid numerical input."
        )