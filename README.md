# Intelligent Healthcare Diagnostic Assistant

An end-to-end AI system that integrates intelligent agents, search, probabilistic reasoning, machine learning, deep learning, NLP, fuzzy logic, and planning into a unified healthcare diagnostic and recommendation platform.


## Project Overview

The Intelligent Healthcare Diagnostic Assistant is an AI-driven decision support system designed to assist users by analyzing symptoms and recommending possible diagnoses and treatment plans.

The project combines multiple Artificial Intelligence techniques into a single intelligent system, allowing different reasoning methods to collaborate before generating the final recommendation.



## 👥 Team Members

| Member | Role |
|--------|------|
| **Tabitha Macharia** | Project Setup + System Integration (app.py) |
| **Esther Kamau** | Intelligent Agent + Knowledge Base |
| **Lorna Kyalo** | Neural Network + Network Planner |
| **Ruth Ndua** | Machine Classifier |
| **Maxwell Chege** | Bayesian Network + Fuzzy Logic |



## System Overview

```text
                               Patient Input
                                      │
                                      ▼
                  Intelligent Agent (Perceive → Think → Act)
                                      │
 ┌────────────────────┬───────────────┬───────────────┬─────────────┐
 │                    │               │               │             │
 ▼                    ▼               ▼               ▼             ▼
Knowledge Base   Fuzzy Logic    Bayesian Net     ML Classifier   Neural Network
                                      │
                                      ▼
                               Treatment Planner
                                      │
                                      ▼
                                Final Diagnosis
```

Each module independently analyzes the patient and returns a diagnosis and confidence score. The Intelligent Agent combines these outputs into a single recommendation.


## Features

- Intelligent symptom collection
- Rule-based reasoning
- Bayesian probabilistic diagnosis
- Machine Learning disease prediction
- Neural Network prediction
- Natural Language Processing
- Fuzzy severity assessment
- AI treatment planning
- Unified diagnosis from multiple AI models


## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow
- Matplotlib
- Git
- GitHub

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/TabbyMacharia/Healthcare-Diagnostic-AI-TeamWatalii.git

cd Healthcare-Diagnostic-AI-TeamWatalii
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
```
**Windows**
```bash
venv\Scripts\activate
```
**Mac / Linux**
```bash
source venv/bin/activate
```
**3. Install dependencies**
```bash
pip install -r requirements.txt
```

## Project Structure

```text
.
├── modules/
│   ├── agent.py
│   ├── bayesian_net.py
│   ├── fuzzy_controller.py
│   ├── knowledge_base.py
│   ├── ml_classifier.py
│   ├── neural_network.py
│   ├── planner.py
│   └── test_ml_classifier.py
├── reports/
│   └── final_report.pdf
├── app.py
├── requirements.txt
├── Environment Setup
├── Kmeans_example_unsupervised_learning_learning_lab_work.ipynb
├── naive bayes example
├── README.md
└── .gitignore
```


## Running the System
Run the full end-to-end system

```bash

python app.py
```

## Evaluation Results

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Decision Tree | Pending | Pending | Pending | Pending |
| Random Forest | Pending | Pending | Pending | Pending |
| Gradient Boosting | Pending | Pending | Pending | Pending |
| Neural Network | Pending | Pending | Pending | Pending |


## Development Workflow

We've set up a branch-based workflow to protect the `main` branch.

### 1. Update your local main branch

```bash
git checkout main
git pull origin main
```

### 2. Create a feature branch

```bash
git checkout -b yourname/feature-name
```

### 3. Make your changes

```bash
git add .
git commit -m "Describe your changes"
```

### 4. Push your branch

```bash
git push origin yourname/feature-name
```

### 5. Open a Pull Request

After pushing your branch, create a Pull Request on GitHub. Once reviewed and approved by the Team Lead, your changes will be merged into `main`.


## License
This project was developed for academic purposes as part of the Artificial Intelligence Capstone Project at Dedan Kimathi University of Technology.



