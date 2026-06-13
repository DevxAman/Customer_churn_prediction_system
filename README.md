# 🎯 Customer Churn Prediction System

An end-to-end Machine Learning application that predicts customer churn using customer demographics, service usage patterns, and account information. The project helps businesses identify high-risk customers and implement proactive retention strategies before customers leave.

---

## 📌 Project Overview

Customer churn is one of the most critical business challenges in industries such as Telecommunications, Banking, SaaS, Insurance, and E-commerce.

This project leverages Machine Learning techniques to:

* Predict whether a customer is likely to churn
* Identify key drivers behind customer attrition
* Provide actionable business recommendations
* Visualize customer behavior through an interactive dashboard
* Support data-driven customer retention strategies

---

## 🚀 Key Features

### Machine Learning Pipeline

* Data Cleaning & Preprocessing
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Feature Selection
* Class Imbalance Handling
* Model Training & Evaluation
* Hyperparameter Optimization
* Explainability Analysis

### Models Implemented

* Logistic Regression
* Random Forest Classifier
* Ensemble Prediction Model

### Business Intelligence

* Churn Risk Segmentation
* Customer Demographic Analysis
* Revenue Impact Assessment
* Retention Strategy Recommendations
* ROI Estimation Dashboard

### Interactive Dashboard

* Real-time Churn Prediction
* KPI Monitoring
* Customer Insights Visualization
* Feature Importance Analysis

---

## 🛠️ Technology Stack

| Category                | Technologies              |
| ----------------------- | ------------------------- |
| Programming Language    | Python 3.9+               |
| Machine Learning        | Scikit-Learn              |
| Data Processing         | Pandas, NumPy             |
| Visualization           | Plotly, Matplotlib        |
| Web Application         | Streamlit                 |
| Model Persistence       | Joblib                    |
| Development Environment | Jupyter Notebook, VS Code |

---

## 📊 Dataset Information

### Dataset

IBM Telco Customer Churn Dataset

### Dataset Summary

| Metric          | Value  |
| --------------- | ------ |
| Total Records   | 7,043  |
| Features        | 21     |
| Target Variable | Churn  |
| Churn Rate      | ~26.5% |

### Features Include

#### Customer Information

* Gender
* Senior Citizen
* Partner
* Dependents

#### Account Information

* Tenure
* Contract Type
* Payment Method
* Paperless Billing

#### Services

* Phone Service
* Internet Service
* Online Security
* Tech Support
* Streaming Services

#### Financial Information

* Monthly Charges
* Total Charges

---

## 🔄 Machine Learning Workflow

### 1. Data Preprocessing

* Missing Value Handling
* Data Type Conversion
* Label Encoding
* Feature Scaling

### 2. Exploratory Data Analysis

* Churn Distribution
* Contract Analysis
* Customer Segmentation
* Correlation Analysis

### 3. Feature Engineering

* Tenure Groups
* Service Count Features
* Customer Risk Categories

### 4. Model Development

* Logistic Regression
* Random Forest
* Ensemble Model

### 5. Model Evaluation

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix

---

## 📈 Model Performance

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 79.5%    | 0.75      | 0.71   | 0.73     | 0.82    |
| Random Forest       | 81.2%    | 0.78      | 0.74   | 0.76     | 0.85    |
| Ensemble Model      | 80.8%    | 0.77      | 0.73   | 0.75     | 0.84    |

---

## 🔍 Important Predictors of Churn

Based on model analysis, the strongest churn indicators are:

1. Contract Type
2. Customer Tenure
3. Payment Method
4. Monthly Charges
5. Senior Citizen Status

### High-Risk Customer Characteristics

* Month-to-Month Contracts
* Less than 12 Months Tenure
* Electronic Check Payments
* Higher Monthly Charges
* No Technical Support Services

---

## 💡 Business Insights

### Key Findings

* Month-to-Month customers exhibit significantly higher churn rates.
* Most customer attrition occurs during the first year.
* Automatic payment users are less likely to churn.
* Long-term contracts substantially improve retention.

### Recommended Actions

#### Immediate Actions

* Promote annual contracts
* Offer onboarding support for new customers
* Incentivize automatic payments

#### Medium-Term Actions

* Develop retention campaigns
* Improve customer engagement programs
* Bundle premium support services

#### Long-Term Actions

* Personalized retention offers using AI
* Customer loyalty programs
* Predictive retention monitoring systems

---

## 📂 Project Structure

```text
Customer-Churn-Prediction-System/
│
├── app.py
├── requirements.txt
├── telco_churn_data.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Feature_Engineering.ipynb
│   └── Model_Training.ipynb
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/DevxAman/Customer_churn_prediction_system.git
cd Customer_churn_prediction_system
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app_final.py
```

Application will start at:

```text
http://localhost:8501
```

---

## 🖥️ Dashboard Features

### Executive Dashboard

* Churn Rate Monitoring
* Revenue Metrics
* Customer Segmentation

### Prediction Center

* Individual Customer Predictions
* Churn Probability Score
* Risk Classification

### Analytics Hub

* Customer Demographics
* Contract Analysis
* Revenue Impact Visualization

### Insights Engine

* Business Recommendations
* Retention Strategies
* ROI Estimation

---

## 🚀 Deployment

### Streamlit Cloud

1. Push repository to GitHub
2. Connect repository to Streamlit Cloud
3. Deploy application

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit","run","app_final.py"]
```

---



---

## 🎓 Learning Outcomes

Through this project:

* Applied supervised machine learning techniques
* Built production-ready ML pipelines
* Developed business-focused analytics solutions
* Created interactive data applications
* Learned model evaluation and deployment workflows

---

## 👨‍💻 Author

**Amandeep Singh**

* B.Tech Computer Science & Engineering
* AI/ML Enthusiast
* Full Stack Developer
* Data Analytics & Machine Learning Practitioner

GitHub: https://github.com/DevxAman

---

## 📜 License

This project is licensed under the MIT License.

---

## ⭐ Support

If you found this project useful, consider giving the repository a star.

It helps showcase the project to recruiters and the developer community.
