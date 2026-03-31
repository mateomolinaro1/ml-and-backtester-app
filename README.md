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