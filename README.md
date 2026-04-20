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

The goal of the project is to build and evaluate an **investment strategy based on macroeconomic signals**.

Concretely, the application:
- collects macroeconomic and financial data
- builds features to capture market dynamics
- trains machine learning models to predict asset behavior
- generates **dynamic portfolio allocations**
- evaluates performance through a **backtesting framework**

---

### Initial state
The original project already included:
- a data pipeline (macro + market data)
- a dynamic allocation strategy
- a backtesting framework

### What we added

This version focuses on **production and MLOps**:

- reproducible environment (uv)
- automated tests (pytest)
- CI pipeline (GitHub Actions)
- Docker containerization + automated delivery (DockerHub)
- experiment tracking with **MLflow**
- web interface/dashboard deployed via **Render**

The goal is to move from a research project to a **deployable ML application with monitoring and interface**.

---

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