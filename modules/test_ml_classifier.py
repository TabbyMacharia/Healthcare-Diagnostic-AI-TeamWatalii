# ============================================================
# test_ml_classifier.py
# Tests the Machine Learning Diagnostic Classifier
# ============================================================

from ml_classifier import MLDiagnosticClassifier
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    print("=" * 60)
    print("ML DIAGNOSTIC CLASSIFIER TEST")
    print("=" * 60)

    # Create classifier
    classifier = MLDiagnosticClassifier()

    # Train the model
    classifier.train()

    # Test patient
    symptoms = [
        "fever",
        "cough",
        "fatigue"
    ]

    # Predict disease
    result = classifier.predict(symptoms)

    print("\nPrediction")
    print("-" * 40)
    print("Symptoms  :", ", ".join(symptoms))
    print("Disease   :", result["diagnosis"])
    print("Confidence:", f"{result['confidence']:.2%}")
    print("Model     :", result["model_used"])

    print("\nTop 5 Predictions")
    print("-" * 40)

    for disease, probability in result["top5"]:
        print(f"{disease:<20} {probability:.2%}")

    print("\nGenerating evaluation graph...")
    print("Close the graph window to end the program.")

    # Show evaluation graph ONCE
    classifier.plot_evaluation()

    print("\nProgram finished successfully.")


if __name__ == "__main__":
    main()