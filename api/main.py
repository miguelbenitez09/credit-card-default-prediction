from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Any

# Crear la aplicación web con FastAPI
app = FastAPI(
    title="API de Predicción de Default de Tarjetas de Crédito",
    description="API para predecir el riesgo de default en pagos de tarjetas de crédito usando modelos de machine learning",
    version="1.0.0"
)

# Cargar el modelo y el escalador al iniciar la aplicación
# Intentar primero la ruta local, luego la de Docker
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')

# Si no existe la ruta local, intentar la ruta de Docker
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join('/app', 'models', 'best_model.pkl')
    SCALER_PATH = os.path.join('/app', 'models', 'scaler.pkl')

try:
    # Intentar cargar los archivos del modelo
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Modelo y escalador cargados correctamente")
except FileNotFoundError as e:
    print(f"❌ Error: No se encontraron los archivos del modelo: {e}")
    model = None
    scaler = None

# Definir el esquema de entrada para las predicciones
class ClientData(BaseModel):
    LIMIT_BAL: float
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float
    utilization_ratio: float
    avg_bill_amt: float
    max_payment_delay: int
    debt_growth: float
    SEX_MAN: int
    SEX_WOMAN: int
    EDUCATION_GRAD_SCHOOL: int
    EDUCATION_HIGH_SCHOOL: int
    EDUCATION_OTHERS: int
    EDUCATION_UNIVERSITY: int
    MARRIAGE_MARRIED: int
    MARRIAGE_OTHERS: int
    MARRIAGE_SINGLE: int

    model_config = {
        "json_schema_extra": {
            "example": {
                "LIMIT_BAL": 50000.0,
                "AGE": 35,
                "PAY_0": 0,
                "PAY_2": 0,
                "PAY_3": 0,
                "PAY_4": 0,
                "PAY_5": 0,
                "PAY_6": 0,
                "BILL_AMT1": 25000.0,
                "BILL_AMT2": 24000.0,
                "BILL_AMT3": 23000.0,
                "BILL_AMT4": 22000.0,
                "BILL_AMT5": 21000.0,
                "BILL_AMT6": 20000.0,
                "PAY_AMT1": 2000.0,
                "PAY_AMT2": 1800.0,
                "PAY_AMT3": 1600.0,
                "PAY_AMT4": 1400.0,
                "PAY_AMT5": 1200.0,
                "PAY_AMT6": 1000.0,
                "utilization_ratio": 0.5,
                "avg_bill_amt": 22500.0,
                "max_payment_delay": 0,
                "debt_growth": -2500.0,
                "SEX_MAN": 0,
                "SEX_WOMAN": 1,
                "EDUCATION_GRAD_SCHOOL": 0,
                "EDUCATION_HIGH_SCHOOL": 0,
                "EDUCATION_OTHERS": 0,
                "EDUCATION_UNIVERSITY": 1,
                "MARRIAGE_MARRIED": 1,
                "MARRIAGE_OTHERS": 0,
                "MARRIAGE_SINGLE": 0
            }
        }
    }

@app.get("/")
def read_root():
    """Página principal que muestra información básica de la API"""
    return {
        "message": "API de Predicción de Default de Tarjetas de Crédito",
        "version": "1.0.0",
        "status": "active" if model is not None else "model_not_loaded",
        "endpoints": {
            "GET /": "Información de la API",
            "GET /health": "Verificación de salud",
            "POST /predict": "Predicción de default",
            "GET /docs": "Documentación interactiva"
        }
    }

@app.get("/health")
def health_check():
    """Verificar si la API y el modelo están funcionando correctamente"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.post("/predict")
def predict_default(client_data: ClientData):
    """
    Recibir datos de un cliente y devolver una predicción de riesgo de default

    Args:
        client_data: Información del cliente en formato JSON

    Returns:
        dict: Resultado con predicción, probabilidad y recomendación
    """
    # Verificar que el modelo esté disponible
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Servicio no disponible: Modelo no cargado"
        )

    try:
        # Convertir los datos recibidos a un DataFrame de pandas
        data_dict = client_data.dict()
        df = pd.DataFrame([data_dict])

        # Aplicar el escalado de datos usando el scaler entrenado
        X_scaled = scaler.transform(df)

        # Usar el modelo para hacer la predicción
        prediction = int(model.predict(X_scaled)[0])
        probability = float(model.predict_proba(X_scaled)[0][1])

        # Determinar el nivel de riesgo basado en la probabilidad
        risk_level = "ALTO" if probability > 0.5 else "BAJO"

        # Generar una recomendación basada en la predicción
        if prediction == 1:
            recommendation = "RECHAZAR SOLICITUD: Alto riesgo de default"
        elif probability > 0.3:
            recommendation = "REVISAR MANUALMENTE: Riesgo moderado"
        else:
            recommendation = "APROBAR SOLICITUD: Bajo riesgo de default"

        # Devolver el resultado completo
        return {
            "prediction": prediction,
            "prediction_label": "DEFAULT" if prediction == 1 else "NO_DEFAULT",
            "probability_default": round(probability, 4),
            "risk_level": risk_level,
            "recommendation": recommendation,
            "confidence": f"{abs(probability - 0.5) * 200:.1f}%"
        }

    except Exception as e:
        # Manejar errores y devolver mensaje claro
        raise HTTPException(
            status_code=400,
            detail=f"Error en la predicción: {str(e)}"
        )

@app.get("/model-info")
def get_model_info():
    """Obtener información técnica sobre el modelo cargado"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    return {
        "model_type": type(model).__name__,
        "model_params": model.get_params() if hasattr(model, 'get_params') else "No disponible",
        "scaler_type": type(scaler).__name__,
        "scaler_params": scaler.get_params() if hasattr(scaler, 'get_params') else "No disponible"
    }

# Código para ejecutar la aplicación cuando se corre directamente
if __name__ == "__main__":
    import uvicorn
    # Iniciar el servidor web en el puerto 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)