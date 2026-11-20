# 🧠 Brain Stroke Prediction Agent

An educational machine-learning application for analyzing patient records, classifying historical stroke labels, and explaining model predictions.

## Features

- Patient assessment and stroke classification
- Multiple machine-learning models
- Model performance analytics
- Global SHAP feature importance
- Individual patient SHAP explanations
- AI-generated model explanations
- Patient assessment history using SQLite
- Individual PDF report generation
- CSV history export

## Machine Learning Algorithms

- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (MLP)

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- SQLite
- Matplotlib
- ReportLab

## Dataset

[Stroke Prediction Dataset – Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

## Project Structure

```text
Brain-Stroke-Prediction/
├── app.py
├── requirements.txt
├── data/
├── models/
├── results/
├── src/
├── README.md
└── .gitignore
```

## Installation and Execution

Clone the repository:

```bash
git clone https://github.com/smiley2805/brain-stroke-prediction-agent.git
cd brain-stroke-prediction-agent
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows activation:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Disclaimer

This project is intended for educational purposes. It classifies historical dataset labels and is not a medical diagnosis or a clinically validated prediction of future stroke risk. SHAP explanations describe model behavior, not medical causation.