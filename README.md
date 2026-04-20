# ML and Backtester App (MLOps version)

## Team
- Molinaro Mateo  
- Chaudron Lucien  
- Aluch Yasmine  
- Li Romain
- Saimane Nawal  

---

## Project overview

This project is based on the following initial work:  
https://github.com/mateomolinaro1/dynamic-allocation-macro-fmp  

We reuse this project as a starting point and extend it in an MLOps perspective.

---

## What the project does

The goal of the project is twofolds: 1) building a machine learning based quantitative (dynamic allocation) strategy
using macroeconomic data (see the report in the original repo "Dynamic Allocation using macro FMPs") and 2) a backtester
engine to evaluate simple univariate ranking strategies based on market and fundamental data from WRDS (CRSP).

Concretely, the application:
- collects macroeconomic and financial data
- builds macro factor mimicking portfolios (FMP)
- trains machine learning models to predict 1-month ahead the selected macro variable
- generates **dynamic portfolio allocations**
- evaluates performance through a **backtesting framework**
- Offers an additional backtesting engine sheet (unrelated to (1)) that allows to backtest simple univariate
ranking strategies
---

### Initial state
The original project already included:
- a data pipeline (macro + market data)
- a dynamic allocation strategy
- a backtesting framework

### What we added

This version focuses on **production and MLOps**:

- live data pipeline for macro data (cannot do it for market data as it requires the IBKR API!)
- reproducible environment (uv)
- automated tests (pytest)
- CI pipeline (GitHub Actions)
- Docker containerization + automated delivery (DockerHub) + automatic app deployment (Render)
- experiment tracking with **MLflow**
- web interface/dashboard deployed via **Render**

The goal is to move from a research project to a **deployable ML application with monitoring and interface**.

---
Before running the project locally, follow the setup steps below to configure your environment properly.

## 0. Folder location
Go to the folder where you want to clone the project
```env
cd /path/to/your/desired/ROOT
```

## 1. Create a `.env` file

Before running the project, create a `.env` file at the **root of the repository**.


The `.env` file should contain the following environment variables (ask me by email for the access tokens at mateo0609@hotmail.fr):
```env
WRDS_USERNAME=
WRDS_PASSWORD=

AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
```

## 2. UV
Check if uv is installed. uv is needed to manage virtual environments with uv venv. If it’s already installed, this
step will skip installing it again.
```env
pip show uv || pip install uv
```

## 3. Clone the project
```env
git clone https://github.com/mateomolinaro1/ml-and-backtester-app.git
cd ml-and-backtester-app
```

## 4. Create a virtual environment using uv
Create a virtual environment named 'venv'
```env
uv venv create venv
```

## 5. Activate the virtual environment
Linux / macOS
```env
source venv/bin/activate
```
Windows (PowerShell)
```env
.\venv\Scripts\Activate.ps1
```

## 6. Install all dependencies
```env
uv install
```

## 7. Run the dashboard
There is an online version available at https://ml-and-backtester-app-latest.onrender.com/ or run it locally with:
```env
uv run --directory src/ml_and_backtester_app/dashboard python app.py
```
and access it at http://localhost:8050/ in your browser.