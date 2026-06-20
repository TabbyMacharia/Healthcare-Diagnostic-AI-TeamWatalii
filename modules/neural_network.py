# ============================================================
# MODULE 5: Deep Neural Network Diagnostic Model
# Covers: Week 10 (Neural Networks)
# ============================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt

class NeuralDiagnosticModel:
    """
    Deep Neural Network for medical diagnosis.
    Architecture: Input → Dense → BN → Dropout → Output
    """

    SYMPTOM_FEATURES = [
        'fever', 'cough', 'fatigue', 'headache',
        'body_aches', 'loss_of_smell', 'chest_pain',
        'rash', 'joint_pain', 'shortness_of_breath',
        'sweating', 'frequent_urination', 'excessive_thirst',
        'blurred_vision', 'night_sweats', 'weight_loss',
        'stiff_neck', 'light_sensitivity'
    ]

    DISEASE_LABELS = [
        'flu', 'covid19', 'dengue', 'cardiac_event',
        'diabetes', 'common_cold', 'tuberculosis', 'meningitis'
    ]

    def __init__(self):
        self.model      = None
        self.history    = None
        self.is_trained = False
        self._build_model()

    def _build_model(self):
        """Build deep MLP architecture"""
        n_inputs  = len(self.SYMPTOM_FEATURES)
        n_outputs = len(self.DISEASE_LABELS)

        self.model = models.Sequential([
            layers.Input(shape=(n_inputs,)),

            # Block 1
            layers.Dense(128, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Block 2
            layers.Dense(64, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            # Block 3
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),

            # Output
            layers.Dense(n_outputs, activation='softmax')
        ], name='MedicalDNN')

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def _generate_data(self, n: int = 3000):
        """Generate synthetic training data"""
        from sklearn.preprocessing import LabelEncoder
        np.random.seed(42)

        profiles = {
            'flu':           {'fever':0.90,'cough':0.85,'fatigue':0.88,
                              'headache':0.70,'body_aches':0.80},
            'covid19':       {'fever':0.88,'cough':0.80,'fatigue':0.90,
                              'loss_of_smell':0.85,'headache':0.65},
            'dengue':        {'fever':0.98,'rash':0.75,'joint_pain':0.85,
                              'headache':0.90,'fatigue':0.80},
            'cardiac_event': {'chest_pain':0.92,'shortness_of_breath':0.88,
                              'sweating':0.75,'fatigue':0.70},
            'diabetes':      {'fatigue':0.82,'frequent_urination':0.95,
                              'excessive_thirst':0.92,'blurred_vision':0.70},
            'common_cold':   {'cough':0.90,'fever':0.50,'headache':0.60,
                              'fatigue':0.55},
            'tuberculosis':  {'cough':0.95,'weight_loss':0.85,'night_sweats':0.80,
                              'fatigue':0.88,'fever':0.70},
            'meningitis':    {'headache':0.95,'stiff_neck':0.90,'fever':0.92,
                              'light_sensitivity':0.85},
        }

        X_list, y_list = [], []
        n_per = n // len(profiles)

        for label_idx, (disease, probs) in enumerate(profiles.items()):
            for _ in range(n_per):
                row = np.array([
                    1 if (np.random.random() <
                          probs.get(feat, 0.03)) else 0
                    for feat in self.SYMPTOM_FEATURES
                ], dtype=np.float32)
                X_list.append(row)
                y_list.append(label_idx)

        X = np.array(X_list)
        y = np.array(y_list)
        idx = np.random.permutation(len(X))
        return X[idx], y[idx]

    def train(self, epochs: int = 50, verbose: int = 1) -> Dict:
        """Train the neural network"""
        X, y = self._generate_data(3000)
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        cb_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy', patience=10,
                restore_best_weights=True),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=5, min_lr=1e-6)
        ]

        print("=" * 55)
        print("  Neural Network — Medical Diagnosis Training")
        print(f"  Architecture: {len(self.SYMPTOM_FEATURES)} → "
              f"128 → 64 → 32 → {len(self.DISEASE_LABELS)}")
        print("=" * 55)
        self.model.summary()

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=64,
            callbacks=cb_list, verbose=verbose
        )

        val_acc = max(self.history.history['val_accuracy'])
        self.is_trained = True
        print(f"\n✅ Best Validation Accuracy: {val_acc:.4f}")
        return {'val_accuracy': val_acc}

    def predict(self, symptoms: List[str]) -> Dict:
        """Predict from symptom list"""
        if not self.is_trained:
            self.train(verbose=0)

        features = np.array([
            [1.0 if feat in [s.lower().replace(' ','_')
                             for s in symptoms]
             else 0.0
             for feat in self.SYMPTOM_FEATURES]
        ], dtype=np.float32)

        proba     = self.model.predict(features, verbose=0)[0]
        pred_idx  = np.argmax(proba)
        diagnosis = self.DISEASE_LABELS[pred_idx]

        return {
            'diagnosis':  diagnosis,
            'confidence': round(float(proba[pred_idx]), 4),
            'all_probs':  dict(zip(self.DISEASE_LABELS,
                                   proba.round(4).tolist()))
        }

    def analyze(self, percept) -> Dict:
        """Module interface for the agent"""
        result = self.predict(percept.symptoms)
        result['summary'] = (f"DNN: {result['diagnosis']} "
                             f"({result['confidence']:.2%})")
        return result

    def plot_training(self):
        """Plot training history"""
        if not self.history:
            print("Train model first!")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        metrics = [('accuracy', 'val_accuracy', 'Accuracy'),
                   ('loss',     'val_loss',     'Loss')]
        colors  = [('#3498db','#e74c3c'), ('#2ecc71','#e67e22')]

        for ax, (train_m, val_m, title), (tc, vc) in zip(
                axes, metrics, colors):
            ax.plot(self.history.history[train_m],
                    color=tc, linewidth=2, label='Train')
            ax.plot(self.history.history[val_m],
                    color=vc, linewidth=2,
                    linestyle='--', label='Validation')
            ax.set_title(f"Model {title}",
                         fontsize=13, fontweight='bold')
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.legend(); ax.grid(True, alpha=0.3)

        plt.suptitle("Neural Network Training Curves",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("nn_training.png", dpi=150)
        plt.show()
