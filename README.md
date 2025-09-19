# Streamlit App — Windows Setup with Poetry

This README shows how to create and use a **local Poetry virtual environment in the project folder** and how to run the Streamlit app.

---

## Project layout

```
.
├─ data/                # Datasets and artifacts
├─ pages/               # Extra Streamlit pages (multipage app)
├─ .gitignore
├─ pyproject.toml       # Poetry configuration (dependencies, scripts)
├─ README.md
└─ streamlit_app.py     # Main Streamlit entrypoint
```

---

## 1) Prerequisites

* **Windows 10/11**
* **Python** (match the version in `pyproject.toml`)
* **PowerShell** or **Command Prompt**
* **Git** (optional, but recommended)

#### Check Python:
```powershell
python --version
where python
```

---

## 2) Install Poetry

Official installer (PowerShell):

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

Add Poetry to PATH if the installer doesn’t do it automatically (you may need to restart your terminal):

```powershell
$env:Path += ";$env:APPDATA\Python\Scripts"
```

Verify:

```powershell
poetry --version
```

---

### 2.1) Configure Poetry to create the venv **inside** the project

> This ensures `.venv/` is created in the same folder as `pyproject.toml`.

```powershell
poetry config virtualenvs.in-project true
```

---

### 2.2) Install dependencies

From the project root (where `pyproject.toml` lives):

```powershell
poetry install
```

Poetry will:

* Create `.venv/` in the project folder (if not already present),
* Resolve and install dependencies defined in `pyproject.toml` (and lock them in `poetry.lock`).

---

### 2.3) Activate the virtual environment (manual)

**Activate script** (manual):

```powershell
.\.venv\Scripts\activate
```

To deactivate later: `deactivate`

---

## 3) Run the Streamlit app locally

From the activated environment:

```powershell
streamlit run streamlit_app.py
```

Streamlit will print a local URL, e.g. `http://localhost:8501`.
Open it in your browser if not opened automatically.

### Multipage notes

* Any `.py` files inside the `pages/` folder become additional pages automatically.
* Use the sidebar page selector in the running app.
