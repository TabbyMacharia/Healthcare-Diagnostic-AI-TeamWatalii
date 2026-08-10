"""
============================================================
MODULE 4: Machine Learning Diagnostic Classifier
Healthcare Diagnostic Assistant

Author: Team Watalii

Description:
This module trains several supervised machine learning
models using patient symptom data and automatically
selects the best-performing model for diagnosis.

Algorithms
----------
• Decision Tree
• Random Forest
• Gradient Boosting

Outputs
-------
• Trained model (.pkl)
• Metrics CSV
• Confusion Matrix
• Feature Importance Graph
============================================================
"""

import os
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report as sklearn_classification_report
)


class MLDiagnosticClassifier:
    """
    Supervised Machine Learning Diagnostic Classifier.

    Trains multiple models and automatically chooses
    the highest-performing classifier.
    """

    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    DATASET_PATH = os.path.join(
        BASE_DIR,
        "data",
        "patient_records.csv"
    )

    MODEL_PATH = os.path.join(
        BASE_DIR,
        "models",
        "ml_classifier.pkl"
    )

    REPORT_FOLDER = os.path.join(
        BASE_DIR,
        "reports"
    )

    FEATURE_COLUMNS = [

        "Fever",
        "Cough",
        "Headache",
        "Fatigue",
        "Sore_Throat",
        "Chest_Pain",
        "Shortness_of_Breath",
        "Nausea",
        "Vomiting",
        "Diarrhea",
        "Body_Ache",
        "Runny_Nose",
        "Sneezing",
        "Loss_of_Taste",
        "Loss_of_Smell",
        "Chills",
        "Dizziness",
        "High_Blood_Pressure",
        "Low_Blood_Pressure",
        "High_Heart_Rate"

    ]

    TARGET_COLUMN = "Diagnosis"

    def __init__(self):

        os.makedirs(
            os.path.dirname(self.MODEL_PATH),
            exist_ok=True
        )

        os.makedirs(
            self.REPORT_FOLDER,
            exist_ok=True
        )

        self.models = {

            "Decision Tree":

                DecisionTreeClassifier(
                    criterion="entropy",
                    random_state=42
                ),

            "Random Forest":

                RandomForestClassifier(
                    n_estimators=150,
                    random_state=42
                ),

            "Gradient Boosting":

                GradientBoostingClassifier(
                    random_state=42
                )

        }

        self.label_encoder = LabelEncoder()

        self.best_model = None
        self.best_model_name = None

        self.is_trained = False

        self.X_test = None
        self.y_test = None

    # ---------------------------------------------------------
    # DATASET LOADING
    # ---------------------------------------------------------

    def load_dataset(self):
        """
        Load patient_records.csv and validate its contents.
        """

        if not os.path.exists(self.DATASET_PATH):

            raise FileNotFoundError(
                f"\nDataset not found:\n{self.DATASET_PATH}"
            )

        df = pd.read_csv(self.DATASET_PATH)

        df.fillna(0, inplace=True)

        missing = [
            col
            for col in self.FEATURE_COLUMNS
            if col not in df.columns
        ]

        if missing:

            raise ValueError(
                f"\nDataset is missing columns:\n{missing}"
            )

        if self.TARGET_COLUMN not in df.columns:

            raise ValueError(
                "Diagnosis column not found."
            )

        return df

    # ---------------------------------------------------------
    # PREPARE DATA
    # ---------------------------------------------------------

    def prepare_data(self):

        df = self.load_dataset()

        X = df[self.FEATURE_COLUMNS]

        y = self.label_encoder.fit_transform(
            df[self.TARGET_COLUMN]
        )

        return train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    # ---------------------------------------------------------
    # TRAIN MODELS
    # ---------------------------------------------------------

    def train(self, verbose=True):
        """
        Train all machine learning models and
        automatically select the best performer.
        """

        X_train, X_test, y_train, y_test = self.prepare_data()

        self.X_test = X_test
        self.y_test = y_test

        results = []

        best_accuracy = 0.0

        if verbose:

            print("\n" + "=" * 65)
            print("TRAINING MACHINE LEARNING DIAGNOSTIC CLASSIFIERS")
            print("=" * 65)

        for model_name, model in self.models.items():

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            precision = precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=3,
                scoring="accuracy"
            )

            results.append({

                "Model": model_name,

                "Accuracy": round(accuracy, 4),

                "Precision": round(precision, 4),

                "Recall": round(recall, 4),

                "F1 Score": round(f1, 4),

                "Cross Validation":
                    round(cv_scores.mean(), 4)

            })

            if verbose:

                print(f"\n{model_name}")

                print("-" * 40)

                print(f"Accuracy         : {accuracy:.4f}")

                print(f"Precision        : {precision:.4f}")

                print(f"Recall           : {recall:.4f}")

                print(f"F1 Score         : {f1:.4f}")

                print(
                    f"Cross Validation : "
                    f"{cv_scores.mean():.4f}"
                )

            if accuracy > best_accuracy:

                best_accuracy = accuracy

                self.best_model = model

                self.best_model_name = model_name

        metrics = pd.DataFrame(results)

        metrics.to_csv(

            os.path.join(

                self.REPORT_FOLDER,

                "ml_metrics.csv"

            ),

            index=False

        )

        joblib.dump(

            {

                "model": self.best_model,

                "encoder": self.label_encoder,

                "model_name": self.best_model_name

            },

            self.MODEL_PATH

        )

        self.is_trained = True

        if verbose:

            print("\n" + "=" * 65)

            print(
                f"BEST MODEL : {self.best_model_name}"
            )

            print(
                f"Accuracy   : {best_accuracy:.4f}"
            )

            print("=" * 65)

            print("\nModel saved successfully.")

            print(
                f"Saved to: {self.MODEL_PATH}"
            )

            print(
                f"Metrics : "
                f"{self.REPORT_FOLDER}/ml_metrics.csv"
            )

        return metrics


    # ---------------------------------------------------------
    # LOAD SAVED MODEL
    # ---------------------------------------------------------

    def load_model(self):
        """
        Load the previously trained model from disk.
        """

        if not os.path.exists(self.MODEL_PATH):

            return False

        saved = joblib.load(self.MODEL_PATH)

        self.best_model = saved["model"]

        self.label_encoder = saved["encoder"]

        self.best_model_name = saved["model_name"]

        self.is_trained = True

        return True
    # ---------------------------------------------------------
    # PREDICT DIAGNOSIS
    # ---------------------------------------------------------

    def predict(self, symptoms: List[str]) -> Dict:
        """
        Predict the most likely diagnosis from
        a list of symptoms.

        Example
        -------
        ["fever", "cough", "fatigue"]
        """

        if not self.is_trained:

            if not self.load_model():

                self.train(verbose=False)

        # ---------------------------------------------
        # Normalize symptom names
        # ---------------------------------------------

        normalized_symptoms = {

            symptom.strip()
                   .lower()
                   .replace(" ", "_")

            for symptom in symptoms

        }

        feature_vector = []

        for feature in self.FEATURE_COLUMNS:

            feature_name = feature.lower()

            if feature_name in normalized_symptoms:

                feature_vector.append(1)

            else:

                feature_vector.append(0)

        feature_vector = np.array(
            feature_vector
        ).reshape(1, -1)

        # ---------------------------------------------
        # Predict diagnosis
        # ---------------------------------------------

        prediction = self.best_model.predict(
            feature_vector
        )[0]

        probabilities = self.best_model.predict_proba(
            feature_vector
        )[0]

        diagnosis = self.label_encoder.inverse_transform(

            [prediction]

        )[0]

        confidence = float(np.max(probabilities))

        probability_table = {}

        for disease, probability in zip(

            self.label_encoder.classes_,

            probabilities

        ):

            probability_table[disease] = round(

                float(probability),

                4

            )

        ranked_predictions = sorted(

            probability_table.items(),

            key=lambda item: item[1],

            reverse=True

        )

        return {

            "diagnosis": diagnosis,

            "confidence": round(confidence, 4),

            "model_used": self.best_model_name,

            "symptom_vector": feature_vector.tolist()[0],

            "top_predictions": ranked_predictions

        }


    # ---------------------------------------------------------
    # AGENT INTERFACE
    # ---------------------------------------------------------

    def analyze(self, percept):
        """
        Interface used by app.py

        percept.symptoms should contain
        a list of patient symptoms.
        """

        result = self.predict(

            percept.symptoms

        )

        result["summary"] = (

            f"{result['model_used']} predicts "

            f"{result['diagnosis']} "

            f"with "

            f"{result['confidence']:.2%} confidence."

        )

        return result


    # ---------------------------------------------------------
    # CLASSIFICATION REPORT
    # ---------------------------------------------------------

    def generate_classification_report(self):
        """
        Print a detailed sklearn classification report.
        """

        if not self.is_trained:

            self.train(verbose=False)

        predictions = self.best_model.predict(

            self.X_test

        )

        print(

            sklearn_classification_report(

                self.y_test,

                predictions,

                target_names=self.label_encoder.classes_

            )

        )


    # ---------------------------------------------------------
    # TOP PREDICTIONS
    # ---------------------------------------------------------

    def print_top_predictions(
            self,
            prediction_result: Dict
    ):
        """
        Display ranked diagnosis predictions.
        """

        print("\nTop Predictions")

        print("-" * 45)

        for disease, probability in prediction_result[
            "top_predictions"
        ]:

            print(

                f"{disease:<25}"

                f"{probability:.2%}"

            )
    # ---------------------------------------------------------
    # PLOT MODEL EVALUATION
    # ---------------------------------------------------------

    def plot_evaluation(self):
        """
        Generate evaluation plots and save them
        inside the reports folder.
        """

        if not self.is_trained:
            self.train(verbose=False)

        predictions = self.best_model.predict(self.X_test)

        labels = self.label_encoder.classes_

        # -------------------------------------------------
        # Confusion Matrix
        # -------------------------------------------------

        cm = confusion_matrix(
            self.y_test,
            predictions
        )

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels
        )

        plt.title(
            f"Confusion Matrix ({self.best_model_name})",
            fontsize=14,
            fontweight="bold"
        )

        plt.xlabel("Predicted Diagnosis")
        plt.ylabel("Actual Diagnosis")

        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        plt.tight_layout()

        confusion_path = os.path.join(
            self.REPORT_FOLDER,
            "confusion_matrix.png"
        )

        plt.savefig(
            confusion_path,
            dpi=300
        )

        plt.show()

        plt.close()

        # -------------------------------------------------
        # Feature Importance
        # -------------------------------------------------

        if hasattr(self.best_model, "feature_importances_"):

            importance = self.best_model.feature_importances_

            importance_df = pd.DataFrame({

                "Feature": self.FEATURE_COLUMNS,

                "Importance": importance

            })

            importance_df = importance_df.sort_values(

                by="Importance",

                ascending=False

            )

            plt.figure(figsize=(12, 8))

            sns.barplot(

                data=importance_df,

                x="Importance",

                y="Feature"

            )

            plt.title(

                "Feature Importance",

                fontsize=14,

                fontweight="bold"

            )

            plt.tight_layout()

            feature_path = os.path.join(

                self.REPORT_FOLDER,

                "feature_importance.png"

            )

            plt.savefig(

                feature_path,

                dpi=300

            )

            plt.show()

            plt.close()

        print("\nEvaluation complete.")

        print(
            f"Metrics CSV           : {self.REPORT_FOLDER}/ml_metrics.csv"
        )

        print(
            f"Confusion Matrix      : {self.REPORT_FOLDER}/confusion_matrix.png"
        )

        print(
            f"Feature Importance    : {self.REPORT_FOLDER}/feature_importance.png"
        )


# ==========================================================
# TESTING
# ==========================================================

if __name__ == "__main__":

    classifier = MLDiagnosticClassifier()

    classifier.train()

    print("\n" + "=" * 60)
    print("SAMPLE PREDICTION")
    print("=" * 60)

    symptoms = [

        "Fever",

        "Cough",

        "Fatigue",

        "Loss_of_Smell"

    ]

    result = classifier.predict(symptoms)

    print(f"\nSymptoms  : {symptoms}")

    print(f"Diagnosis : {result['diagnosis']}")

    print(f"Confidence: {result['confidence']:.2%}")

    print(f"Model     : {result['model_used']}")

    classifier.print_top_predictions(result)

    print("\nGenerating classification report...\n")

    classifier.generate_classification_report()

    print("\nGenerating evaluation graphs...\n")

    classifier.plot_evaluation()

    print("\nModule testing completed successfully.")