import streamlit as st
import requests
import json
import pandas as pd
from typing import Dict, Any

# Configurar la página principal de la aplicación web
st.set_page_config(
    page_title="Predicción de Default - Interfaz Web",
    page_icon="💳",
    layout="wide"
)

# Dirección del servidor de la API
# Cuando está en Docker, usar el nombre del servicio
# Cuando está local, usar localhost
import os
API_BASE_URL = "http://credit-default-api:8000" if os.path.exists('/app') else "http://localhost:8000"

def check_api_health() -> Dict[str, Any]:
    """Verificar si la API está funcionando correctamente"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}

def make_prediction(data: Dict[str, Any]) -> Dict[str, Any]:
    """Enviar datos a la API y obtener una predicción"""
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=data)
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def load_json_file(uploaded_file) -> Dict[str, Any]:
    """Leer y validar un archivo JSON subido por el usuario"""
    try:
        data = json.load(uploaded_file)
        return data
    except json.JSONDecodeError as e:
        st.error(f"Error al leer el archivo JSON: {e}")
        return None

def main():
    # Título principal de la aplicación
    st.title("💳 Predicción de Default en Tarjetas de Crédito")
    st.markdown("Interfaz web para probar la API de predicción de riesgo de default")

    # Panel lateral con información y controles
    with st.sidebar:
        st.header("ℹ️ Información")
        st.markdown("""
        Esta interfaz permite:
        - Ingresar datos manualmente
        - Subir archivos JSON con datos
        - Verificar el estado de la API
        - Realizar predicciones
        """)

        # Botón para verificar el estado de la API
        if st.button("🔍 Verificar Estado API"):
            with st.spinner("Verificando..."):
                health = check_api_health()
                if health.get("status") == "healthy":
                    st.success("✅ API funcionando correctamente")
                    st.json(health)
                else:
                    st.error("❌ API no disponible")
                    st.json(health)

    # Crear pestañas para diferentes funcionalidades
    tab1, tab2, tab3 = st.tabs(["📝 Ingreso Manual", "📤 Subir JSON", "📊 Resultados"])

    # Pestaña 1: Formulario para ingresar datos manualmente
    with tab1:
        st.header("Ingreso Manual de Datos")

        # Organizar el formulario en columnas para mejor visualización
        col1, col2, col3 = st.columns(3)

        with col1:
            limit_bal = st.number_input("Límite de Crédito (LIMIT_BAL)", min_value=0.0, value=50000.0)
            age = st.number_input("Edad (AGE)", min_value=18, max_value=100, value=35)
            pay_0 = st.selectbox("Estado Pago Mes 0 (PAY_0)", options=["-1","0","1","2","3","4","5","6","7","8","9"], index=1)
            pay_2 = st.selectbox("Estado Pago Mes 2 (PAY_2)", options=["-1","0","1","2","3","4","5","6","7","8","9"], index=1)
            pay_3 = st.selectbox("Estado Pago Mes 3 (PAY_3)", options=["-1","0","1","2","3","4","5","6","7","8","9"], index=1)

        with col2:
            pay_4 = st.selectbox("Estado Pago Mes 4 (PAY_4)", options=["-1","0","1","2","3","4","5","6","7","8","9"], index=1)
            pay_5 = st.selectbox("Estado Pago Mes 5 (PAY_5)", options=["-1","0","1","2","3","4","5","6","7","8","9"], index=1)
            pay_6 = st.selectbox("Estado Pago Mes 6 (PAY_6)", options=["-1","0","1","2","3","4","5","6","7","8","9"], index=1)
            bill_amt1 = st.number_input("Factura Mes 1 (BILL_AMT1)", value=25000.0)
            bill_amt2 = st.number_input("Factura Mes 2 (BILL_AMT2)", value=24000.0)

        with col3:
            bill_amt3 = st.number_input("Factura Mes 3 (BILL_AMT3)", value=23000.0)
            bill_amt4 = st.number_input("Factura Mes 4 (BILL_AMT4)", value=22000.0)
            bill_amt5 = st.number_input("Factura Mes 5 (BILL_AMT5)", value=21000.0)
            bill_amt6 = st.number_input("Factura Mes 6 (BILL_AMT6)", value=20000.0)
            pay_amt1 = st.number_input("Pago Mes 1 (PAY_AMT1)", value=2000.0)

        # Más campos
        col4, col5, col6 = st.columns(3)

        with col4:
            pay_amt2 = st.number_input("Pago Mes 2 (PAY_AMT2)", value=1800.0)
            pay_amt3 = st.number_input("Pago Mes 3 (PAY_AMT3)", value=1600.0)
            pay_amt4 = st.number_input("Pago Mes 4 (PAY_AMT4)", value=1400.0)

        with col5:
            pay_amt5 = st.number_input("Pago Mes 5 (PAY_AMT5)", value=1200.0)
            pay_amt6 = st.number_input("Pago Mes 6 (PAY_AMT6)", value=1000.0)
            utilization_ratio = st.number_input("Ratio de Utilización", min_value=0.0, max_value=1.0, value=0.5)

        with col6:
            avg_bill_amt = st.number_input("Promedio Facturas", value=22500.0)
            max_payment_delay = st.number_input("Máx Retraso Pago", min_value=0, value=0)
            debt_growth = st.number_input("Crecimiento Deuda", value=-2500.0)

        # Variables categóricas
        st.subheader("Variables Categóricas")
        col7, col8, col9 = st.columns(3)

        with col7:
            sex = st.selectbox("Género", options=["Hombre", "Mujer"], index=1)
            sex_man = 1 if sex == "Hombre" else 0
            sex_woman = 1 if sex == "Mujer" else 0

        with col8:
            education = st.selectbox("Educación", options=["Posgrado", "Universidad", "Secundaria", "Otros"], index=1)
            education_grad = 1 if education == "Posgrado" else 0
            education_uni = 1 if education == "Universidad" else 0
            education_high = 1 if education == "Secundaria" else 0
            education_others = 1 if education == "Otros" else 0

        with col9:
            marriage = st.selectbox("Estado Civil", options=["Casado", "Soltero", "Otros"], index=1)
            marriage_married = 1 if marriage == "Casado" else 0
            marriage_single = 1 if marriage == "Soltero" else 0
            marriage_others = 1 if marriage == "Otros" else 0

        # Botón de predicción
        if st.button("🔮 Realizar Predicción", type="primary"):
            # Preparar datos
            data = {
                "LIMIT_BAL": limit_bal,
                "AGE": age,
                "PAY_0": int(pay_0),
                "PAY_2": int(pay_2),
                "PAY_3": int(pay_3),
                "PAY_4": int(pay_4),
                "PAY_5": int(pay_5),
                "PAY_6": int(pay_6),
                "BILL_AMT1": bill_amt1,
                "BILL_AMT2": bill_amt2,
                "BILL_AMT3": bill_amt3,
                "BILL_AMT4": bill_amt4,
                "BILL_AMT5": bill_amt5,
                "BILL_AMT6": bill_amt6,
                "PAY_AMT1": pay_amt1,
                "PAY_AMT2": pay_amt2,
                "PAY_AMT3": pay_amt3,
                "PAY_AMT4": pay_amt4,
                "PAY_AMT5": pay_amt5,
                "PAY_AMT6": pay_amt6,
                "utilization_ratio": utilization_ratio,
                "avg_bill_amt": avg_bill_amt,
                "max_payment_delay": max_payment_delay,
                "debt_growth": debt_growth,
                "SEX_MAN": sex_man,
                "SEX_WOMAN": sex_woman,
                "EDUCATION_GRAD_SCHOOL": education_grad,
                "EDUCATION_HIGH_SCHOOL": education_high,
                "EDUCATION_OTHERS": education_others,
                "EDUCATION_UNIVERSITY": education_uni,
                "MARRIAGE_MARRIED": marriage_married,
                "MARRIAGE_OTHERS": marriage_others,
                "MARRIAGE_SINGLE": marriage_single
            }

            with st.spinner("Realizando predicción..."):
                result = make_prediction(data)

            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.success("Predicción completada!")
                st.json(result)

                # Mostrar resultados de forma visual
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.metric("Predicción", result["prediction_label"])
                    st.metric("Probabilidad Default", f"{result['probability_default']:.1%}")
                with col_res2:
                    st.metric("Nivel de Riesgo", result["risk_level"])
                    st.info(result["recommendation"])

    with tab2:
        st.header("Subir Archivo JSON")

        # Widget para seleccionar archivo
        uploaded_file = st.file_uploader("Selecciona un archivo JSON", type="json")

        if uploaded_file is not None:
            st.success("Archivo cargado exitosamente!")

            # Mostrar el contenido del archivo
            data = load_json_file(uploaded_file)
            if data:
                st.json(data)

                # Botón para hacer predicción con los datos del archivo
                if st.button("🔮 Realizar Predicción con JSON", type="primary"):
                    with st.spinner("Realizando predicción..."):
                        result = make_prediction(data)

                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.success("Predicción completada!")
                        st.json(result)

                        # Mostrar resultados de forma visual
                        col_res1, col_res2 = st.columns(2)
                        with col_res1:
                            st.metric("Predicción", result["prediction_label"])
                            st.metric("Probabilidad Default", f"{result['probability_default']:.1%}")
                        with col_res2:
                            st.metric("Nivel de Riesgo", result["risk_level"])
                            st.info(result["recommendation"])

    with tab3:
        st.header("Historial de Resultados")
        st.info("Esta sección mostrará el historial de predicciones realizadas (funcionalidad pendiente)")

# Ejecutar la aplicación principal
if __name__ == "__main__":
    main()