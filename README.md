
# Diabetes Prediction API & Frontend (Pima Indians Dataset)

This project trains multiple classifiers on the **Pima Indians Diabetes** dataset, serves predictions via a **FastAPI** backend, containerizes with **Docker**, deploys to **Render**, and includes two frontend options: **Streamlit** and a simple **HTML+JS** page.

> **Note:** The dataset file `diabetes.csv` is **not** included. Download it from Kaggle/UC Irvine and place it at the project root before training.

---

## 📁 Repo Structure

```
diabetes_api_project/
├─ app/
│  ├─ main.py                # FastAPI app (async)
│  ├─ __init__.py
├─ models/
│  ├─ (generated) diabetes_model.pkl
│  ├─ (generated) metrics.json
│  ├─ (generated) feature_order.json
├─ frontend_streamlit/
│  └─ app.py                 # Streamlit frontend
├─ frontend_web/
│  ├─ index.html             # Simple HTML+JS frontend
│  └─ styles.css
├─ train_model.py            # Train/evaluate/save best model
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
├─ render.yaml               # Render deploy spec (optional)
├─ README.md
└─ sample_request.json
```

---

## 🔧 1) Setup & Training

1. Create & activate a virtual environment (example for Windows PowerShell):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   macOS/Linux:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset as `diabetes.csv` (must contain columns):
   `Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome`
   and place it in the project root: `diabetes_api_project/diabetes.csv`

4. Train models and save the best:
   ```bash
   python train_model.py
   ```
   This will generate:
   - `models/diabetes_model.pkl`
   - `models/metrics.json`
   - `models/feature_order.json`

---

## 🚀 2) Run the FastAPI backend (locally)

```bash
uvicorn app.main:app --reload
```
Then visit:
- Health check: `http://127.0.0.1:8000/health`
- Docs (Swagger): `http://127.0.0.1:8000/docs`

**POST /predict** sample:
```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @sample_request.json
```

---

## 🧪 3) Streamlit Frontend

```bash
streamlit run frontend_streamlit/app.py
```
Set the API URL in the Streamlit sidebar if deploying remotely (e.g., Render).

---

## 🌐 4) Simple Web Frontend

Open `frontend_web/index.html` in a live server (e.g., VS Code Live Server).  
Set the backend URL at the top of the HTML file if your API is remote.

---

## 🐳 5) Docker

Build image (after training so the model files exist):
```bash
docker build -t diabetes-api .
```

Run container:
```bash
docker run -p 8000:8000 diabetes-api
```

Optional Compose:
```bash
docker-compose up --build
```

---

## ☁️ 6) Deploy to Render

1. Push this repo to GitHub.
2. Create a new **Web Service** on Render:
   - Runtime: Docker
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
3. Ensure your trained `models/*.pkl` and JSON files are committed or available at build time.

**Note:** If you prefer a non-Docker Render service, use `render.yaml` as a guide and set the Start command accordingly.

---

## ✅ Endpoints Summary

- `GET /health` → `{ "status": "ok" }`
- `POST /predict` (async) → returns `prediction`, `result` (`"Diabetic"`/`"Not Diabetic"`), and `confidence`
- `GET /metrics` → returns evaluation metrics (Accuracy, Precision, Recall, F1) from the test set

---

## 📦 Submission Checklist

- [x] Trained at least 2 classifiers; evaluates multiple
- [x] Evaluation metrics printed & saved
- [x] Best model saved with `joblib`
- [x] FastAPI async endpoints
- [x] Dockerfile included
- [x] Render deployment ready
- [x] Frontend (Streamlit + Web) integrated with API
- [x] Clean folder structure
- [x] Sample request payload

Good luck! ✨
