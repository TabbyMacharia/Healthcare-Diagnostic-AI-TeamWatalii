# ============================================================
# MODULE 3: Bayesian Network — Probabilistic Diagnosis
# Covers: Week 7 (Bayesian Networks)
# ============================================================

import numpy as np
from typing import Dict, List

# Added for real Bayesian Network
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


class SimpleBayesianDiagnostics:
    """
    Bayesian diagnostic model using a real Bayesian Network.
    Original structure preserved.
    """

    def __init__(self):

        # -----------------------------
        # Original priors
        # -----------------------------

        self.priors = {
            'flu':        0.15,
            'covid19':    0.08,
            'dengue':     0.05,
            'cardiac':    0.04,
            'diabetes':   0.10,
            'common_cold':0.30,
            'healthy':    0.28,
        }


        # -----------------------------
        # Original likelihoods preserved
        # -----------------------------

        self.likelihoods = {

            'flu':
            {'fever':.90,'cough':.85,'fatigue':.88,
             'loss_of_smell':.20,'chest_pain':.05},

            'covid19':
            {'fever':.88,'cough':.80,'fatigue':.90,
             'loss_of_smell':.85,'chest_pain':.20},

            'dengue':
            {'fever':.98,'rash':.75,
             'joint_pain':.85,'fatigue':.80},

            'cardiac':
            {'chest_pain':.92,
             'fatigue':.70},

            'diabetes':
            {'fatigue':.82},

            'common_cold':
            {'cough':.90,'fever':.50},

            'healthy':
            {'fever':.02,'cough':.05}

        }


        # -----------------------------
        # Create Bayesian Network
        # -----------------------------

        self.network = DiscreteBayesianNetwork([

            ('Disease','Fever'),
            ('Disease','Cough'),
            ('Disease','Fatigue'),
            ('Disease','Loss_of_smell'),
            ('Disease','Chest_pain'),
            ('Disease','Rash'),
            ('Disease','Joint_pain')

        ])


        diseases=list(self.priors.keys())


        # Disease node CPT

        cpds=[

            TabularCPD(
                variable='Disease',
                variable_card=len(diseases),
                values=[[p] for p in self.priors.values()],
                state_names={
                    'Disease':diseases
                }
            )

        ]


        # -----------------------------
        # Generate symptom CPTs
        # -----------------------------

        symptoms=[

            'fever',
            'cough',
            'fatigue',
            'loss_of_smell',
            'chest_pain',
            'rash',
            'joint_pain'

        ]


        for symptom in symptoms:


            yes=[]


            for disease in diseases:

                yes.append(

                    self.likelihoods
                    .get(disease,{})
                    .get(symptom,0.01)

                )


            cpds.append(

                TabularCPD(

                    variable=symptom.capitalize(),

                    variable_card=2,

                    values=[
                        yes,
                        [1-x for x in yes]
                    ],

                    evidence=['Disease'],

                    evidence_card=[len(diseases)],


                    state_names={

                        symptom.capitalize():
                        ['yes','no'],

                        'Disease':
                        diseases

                    }

                )

            )


        self.network.add_cpds(*cpds)


        if self.network.check_model():

            print("Bayesian Network Created")


        self.inference = VariableElimination(
            self.network
        )


    # =====================================================
    # Bayesian inference replaces old Naive Bayes calculation
    # =====================================================


    def compute_posterior(self,
                          symptoms: List[str]) -> Dict[str,float]:


        evidence={}


        for symptom in symptoms:

            symptom=symptom.lower().replace(" ","_")


            evidence[
                symptom.capitalize()
            ]="yes"



        result=self.inference.query(

            variables=['Disease'],

            evidence=evidence

        )


        return {

            disease:
            round(float(prob),4)

            for disease,prob in zip(

                result.state_names['Disease'],

                result.values

            )

        }



    def analyze(self, percept) -> Dict:


        posteriors=self.compute_posterior(
            percept.symptoms
        )


        top=max(
            posteriors,
            key=posteriors.get
        )


        ranked=sorted(
            posteriors.items(),
            key=lambda x:x[1],
            reverse=True
        )


        return {

            'summary':
            f"Top: {top} ({posteriors[top]:.2%})",

            'diagnosis':top,

            'confidence':
            posteriors[top],

            'all_posteriors':
            posteriors,

            'ranked_diagnoses':
            ranked[:5]

        }



    def explain(self,disease,symptoms):

        return (
            f"Bayesian inference updated probability "
            f"for {disease} using symptoms {symptoms}"
        )
    # ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    diagnostics = SimpleBayesianDiagnostics()

    print("\n=== BAYESIAN DIAGNOSTIC SYSTEM ===")
    print(
        "Examples: fever, cough, fatigue, loss_of_smell, chest_pain, rash"
    )

    user_input = input(
        "\nEnter symptoms separated by commas: "
    )


    symptoms = [
        s.strip()
        for s in user_input.split(",")
        if s.strip()
    ]


    class Percept:

        def __init__(self, symptoms):
            self.symptoms = symptoms


    result = diagnostics.analyze(
        Percept(symptoms)
    )


    print("\n" + "="*45)
    print("        DIAGNOSTIC REPORT")
    print("="*45)


    print(result["summary"])

    print(
        f"Confidence: {result['confidence']:.2%}"
    )


    print("\n--- Ranked Diagnoses ---")

    for rank,(disease,prob) in enumerate(
        result["ranked_diagnoses"],
        start=1
    ):

        print(
            f"{rank}. {disease:<15} {prob:.2%}"
        )