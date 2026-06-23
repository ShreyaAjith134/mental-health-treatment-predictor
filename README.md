# Mental Health Treatment Predictor

A machine learning-based web application that predicts whether an individual is likely to require mental health treatment using workplace and personal factors from the OSMI Mental Health in Tech Survey dataset.

---

## Project Overview

| Attribute       | Details                                |
| --------------- | -------------------------------------- |
| Project Type    | Machine Learning Classification        |
| Domain          | Healthcare Analytics                   |
| Dataset         | OSMI Mental Health in Tech Survey 2014 |
| Records         | 1,259                                  |
| Features Used   | 22                                     |
| Target Variable | Treatment                              |
| Deployment      | Flask Web Application                  |
| Best Model      | Stacking Ensemble                      |
| Language        | Python                                 |

---

## Problem Statement

Mental health issues in the technology sector are often underreported due to stigma and lack of awareness. Organizations typically rely on manual assessments which require significant time and resources.

This project aims to develop a machine learning solution capable of predicting whether an employee is likely to require mental health treatment based on survey responses and workplace factors.

---

## System Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Exploratory Data Analysis
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Stacking Ensemble Selection
   ↓
Flask Deployment
```

---

## Dataset Information

### Source

OSMI Mental Health in Tech Survey (2014)

### Important Features

| Feature          | Description                               |
| ---------------- | ----------------------------------------- |
| Age              | Age of respondent                         |
| Gender           | Standardized gender category              |
| Family History   | Family history of mental illness          |
| Work Interfere   | Impact of mental health on work           |
| Benefits         | Availability of mental health benefits    |
| Care Options     | Awareness of care resources               |
| Wellness Program | Employer wellness initiatives             |
| Supervisor       | Comfort discussing issues with supervisor |
| Coworkers        | Comfort discussing issues with coworkers  |

### Target Variable

| Value | Meaning                    |
| ----- | -------------------------- |
| 0     | Does Not Require Treatment |
| 1     | Requires Treatment         |

---

## Machine Learning Models Evaluated

| Model               |
| ------------------- |
| Logistic Regression |
| K-Nearest Neighbors |
| Decision Tree       |
| Random Forest       |
| Bagging Classifier  |
| AdaBoost Classifier |
| Stacking Classifier |

---

## Final Model

### Stacking Ensemble Classifier

The final deployed model uses a Stacking Ensemble architecture.

| Layer         | Models Used                             |
| ------------- | --------------------------------------- |
| Base Learners | Logistic Regression, KNN, Decision Tree |
| Meta Learner  | Random Forest                           |
| Output        | Treatment Prediction                    |

The ensemble approach combines the strengths of multiple algorithms, resulting in improved prediction performance compared to individual classifiers.

---

## Performance Metrics

The models were evaluated using the following metrics:

| Metric           | Purpose                              |
| ---------------- | ------------------------------------ |
| Accuracy         | Overall correctness                  |
| Precision        | Quality of positive predictions      |
| Recall           | Ability to identify positive cases   |
| F1 Score         | Balance between Precision and Recall |
| MCC              | Balanced evaluation metric           |
| Confusion Matrix | Detailed classification analysis     |

---

## Results

| Metric    | Stacking Ensemble |
| --------- | ----------------- |
| Accuracy  | ~83%              |
| Precision | ~84%              |
| Recall    | ~85%              |
| F1 Score  | ~84%              |
| MCC       | ~0.65             |

The Stacking Ensemble achieved the best overall performance and was selected for deployment.

---

## Application Screenshots

### Home Page

<p align="center">
Insert Screenshot Here
</p>

---

### Prediction Result

<p align="center">
Insert Screenshot Here
</p>

---

### Accuracy Comparison

<p align="center">
Insert Screenshot Here
</p>

---

### Confusion Matrix

<p align="center">
Insert Screenshot Here
</p>

---

## Project Structure

```text
mental-health-treatment-predictor
│
├── templates/
│   └── index.html
│
├── app.py
├── preprocess.py
├── retrain.py
├── evaluate.py
├── manual_test.py
│
├── Logistic_Regression.pkl
├── K_Neighbors_Classifier.pkl
├── Decision_Tree_Classifier.pkl
├── Random_Forests.pkl
├── Bagging.pkl
├── Boosting.pkl
├── Stacking.pkl
│
├── accuracy_comparison.png
├── cm_Boosting.png
├── cm_Stacking.png
│
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/mental-health-treatment-predictor.git
cd mental-health-treatment-predictor
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

---

## Technology Stack

| Category             | Technologies  |
| -------------------- | ------------- |
| Programming Language | Python        |
| Backend              | Flask         |
| Machine Learning     | Scikit-learn  |
| Data Processing      | Pandas, NumPy |
| Visualization        | Matplotlib    |
| Frontend             | HTML, CSS     |
| Version Control      | Git, GitHub   |

---

## Future Enhancements

* Hyperparameter Optimization
* Explainable AI using SHAP
* Cloud Deployment
* Integration of Recent OSMI Datasets
* Interactive Dashboard Development
* Enhanced Privacy and Security Features

---

## Conclusion

The Mental Health Treatment Predictor demonstrates the application of machine learning in workplace mental health analytics. Through extensive model comparison and ensemble learning techniques, the project provides a scalable and data-driven approach to identifying individuals who may benefit from mental health support.

---

## Authors

Developed as an academic machine learning project for mental health treatment prediction.
