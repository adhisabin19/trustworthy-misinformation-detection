## Week 6 — Trustworthy Classification and Application Deployment

TrustFact is an experimental six-class misinformation classification system developed as part of the CPSC 59000 capstone project at Lewis University.

### Research Results

| Metric | Result |
|---|---:|
| Selected model | DistilBERT V1 |
| Validation Macro-F1 | 0.2742 |
| Test accuracy | 29.04% |
| Test Macro-F1 | 0.2777 |
| Validation ECE reduction | 70.61% |
| Test ECE reduction | 84.78% |
| Temperature | 1.429502 |
| Confidence threshold | 0.275 |
| Selective accuracy | 31.09% |
| Test coverage | 59.91% |

### Application Features

- Six-class LIAR statement classification
- Temperature-scaled confidence calibration
- Uncertainty estimation and prediction abstention
- Live SHAP and LIME explanations
- FastAPI REST backend
- Interactive HTML, CSS and JavaScript frontend
- Browser-based prediction history
- Light and dark themes

### Running the Application Locally

Create and activate a Python virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the deployment dependencies:

```powershell
python -m pip install -r requirements-deployment.txt
```

Obtain the trained TrustFact deployment package separately and extract it into the project root. The backend expects the following paths:

```text
TrustFact_Model/
    TrustFact_Model/
        model/
        final_model_config.json
```

Verify local model inference:

```powershell
python test_local_model.py
```

Start the application:

```powershell
python -m uvicorn backend:app --app-dir trustfact_frontend --host 127.0.0.1 --port 8000
```

Frontend: http://127.0.0.1:8000/

API documentation: http://127.0.0.1:8000/docs

### Research Limitations

TrustFact is a research prototype, not an independent fact-checking service. The model uses statement text alone and does not retrieve external factual evidence. Calibration improves confidence reliability but does not improve classification accuracy. Model explanations describe predictive behavior rather than factual correctness.

### Authors

- Sabin Adhikari
- Mohammad Hasan Ali Alshaggah

Lewis University — Fall 2026