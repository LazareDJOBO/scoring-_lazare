from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Charger le modèle, scaler et colonnes
regmodel = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")  # Colonnes après encodage

# Mapping cible
target_map = {0: "Low", 1: "Average", 2: "High"}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    home_ownership: str = Form(...),
    marital_status: str = Form(...),
    education: str = Form(...),
    income: float = Form(...),
    age: float = Form(...)
):
    # Créer le DataFrame
    data = pd.DataFrame([{
        "Home Ownership": home_ownership,
        "Marital Status": marital_status,
        "Education": education,
        "Income": income,
        "Age": age
    }])

    # Encodage identique à l’entraînement
    categorical_features = ["Home Ownership", "Marital Status", "Education"]
    data_encoded = pd.get_dummies(data, columns=categorical_features, drop_first=True)

    # Aligner avec les colonnes du modèle
    data_encoded = data_encoded.reindex(columns=columns, fill_value=0)

    # Normalisation
    data_scaled = scaler.transform(data_encoded)

    # Prédiction
    pred = regmodel.predict(data_scaled)[0]
    credit_score = target_map[pred]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "predicted_score": credit_score
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
