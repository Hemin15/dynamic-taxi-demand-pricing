# 🚖 Dynamic Taxi Demand Forecasting & Pricing Engine

An end-to-end AI-powered system that predicts taxi demand and optimizes pricing strategies to maximize revenue using machine learning and decision intelligence.

---

## 📌 Overview

Urban mobility platforms rely on accurate demand forecasting and adaptive pricing to balance rider demand, driver supply, and profitability. This project builds a production-style analytics pipeline that:

* Forecasts taxi demand using historical trip patterns
* Compares multiple machine learning models
* Selects the best-performing model automatically
* Applies dynamic pricing optimization
* Visualizes decisions through an interactive Streamlit dashboard

The final system combines **predictive analytics + pricing intelligence**.

---

## 🚀 Key Features

### 📈 Demand Forecasting

* Predicts taxi demand by zone and hour
* Uses engineered temporal + geographic features
* Handles real-world noisy transportation data

### 🤖 Machine Learning Model Benchmarking

Compared multiple models including:

* LightGBM
* XGBoost
* Random Forest
* Extra Trees
* HistGradientBoosting

### 💰 Dynamic Pricing Engine

* Rule-based pricing optimization
* Revenue-maximizing price search
* Elasticity sensitivity analysis
* Static vs dynamic revenue comparison

### 📊 Interactive Dashboard

Built using Streamlit:

* Real-time demand prediction
* Recommended pricing output
* Revenue uplift metrics
* Pricing curves
* Model quality reports

---

## 🧠 Tech Stack

| Category        | Tools                           |
| --------------- | ------------------------------- |
| Language        | Python                          |
| Data            | Pandas, NumPy                   |
| ML              | Scikit-learn, LightGBM, XGBoost |
| Visualization   | Matplotlib                      |
| Deployment      | Streamlit                       |
| Version Control | Git + GitHub                    |

---

## 📂 Project Structure

```text
dynamic-taxi-demand-pricing/
│── app/                # Streamlit dashboard
│── src/                # ML pipeline + pricing engine
│── outputs/            # Reports / charts / artifacts
│── README.md
│── requirements.txt
│── .gitignore
```

---

## 📈 Results

### Best Performing Model

**LightGBM**

### Revenue Optimization Impact

* **Average Revenue Uplift:** 22.23%
* **Demo Scenario Uplift:** 19.50%

### Example Scenario

| Metric           | Value   |
| ---------------- | ------- |
| Predicted Demand | 27.92   |
| Static Price     | 38.25   |
| Optimal Price    | 51.69   |
| Revenue Increase | +19.50% |

---

## 🌍 Streamlit Dashboard

Interactive web application for pricing decisions:

```bash
streamlit run app/app.py
```

---

## ⚙️ Installation

Clone repository:

```bash
git clone https://github.com/Hemin15/dynamic-taxi-demand-pricing.git
cd dynamic-taxi-demand-pricing
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run application:

```bash
streamlit run app/app.py
```

---


## 💼 Industry Value

This project demonstrates:

* Machine Learning Engineering
* Forecasting Systems
* Revenue Optimization
* Data Visualization
* Model Evaluation
* Product Deployment

---

## 🔮 Future Enhancements

* Real-time API integration
* Live weather / traffic signals
* Driver supply forecasting
* Reinforcement learning pricing engine
* Cloud deployment pipeline

---

## 👨‍💻 Author

**Hemin Modi**
Computer Engineering Student
Passionate about AI, Data Science, and Intelligent Systems

---


