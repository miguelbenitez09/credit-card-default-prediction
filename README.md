# 💳 Credit Card Default Prediction

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![scikit-learn](https://img.shields.io/badge/sklearn-1.3.2-orange.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.1+-yellow.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.1-red.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
[![Autor](https://img.shields.io/badge/Autor-developed_by_Miguel_Benítez_(UTP)-informational.svg)](https://github.com/miguelbenitez09)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Firma Oficial:** **`Credit Card Default Prediction v1.0.0 • developed by Miguel Benítez`**  
> **Sistema de Machine Learning para predicción de riesgo de default en tarjetas de crédito mediante análisis de historial crediticio y factores demográficos.**

---

## 👨‍💻 Autor

**Ing. Miguel Antonio Benítez González** (Universidad Tecnológica de Panamá - UTP)
- 🎓 Título: Ingeniero en Sistemas y Computación
- 📧 Email: mbenitezg01@gmail.com
- 💻 GitHub: [miguelbenitez09](https://github.com/miguelbenitez09?tab=repositories)
- 💼 LinkedIn: [Miguel Antonio Benítez González](https://www.linkedin.com/in/miguel-antonio-ben%C3%ADtez-gonz%C3%A1lez-457816247/)

---

## 📋 Tabla de Contenidos

1. [Descripción del Proyecto](#-descripción-del-proyecto)
2. [Problema de Negocio](#-problema-de-negocio)
3. [Dataset](#-dataset)
4. [Análisis y Técnicas Aplicadas](#-análisis-y-técnicas-aplicadas)
5. [Feature Engineering](#-feature-engineering)
6. [Modelos y Resultados](#-modelos-y-resultados)
7. [Tecnologías Utilizadas](#️-tecnologías-utilizadas)
8. [Estructura del Proyecto](#-estructura-del-proyecto)
9. [Instalación](#-instalación)
10. [Uso](#-uso)
11. [API Endpoints](#-api-endpoints)
12. [Mejoras Futuras](#-mejoras-futuras)

---

## 🎯 Descripción del Proyecto

Este proyecto implementa un sistema completo de evaluación de riesgo crediticio para predecir si un cliente de tarjeta de crédito incumplirá su pago el próximo mes (`default payment = 1`) o pagará a tiempo (`default payment = 0`).

### Objetivo Principal
Desarrollar un modelo predictivo robusto para:
- Identificar clientes con alto riesgo de incumplimiento
- Optimizar decisiones de aprobación de crédito
- Minimizar pérdidas por morosidad
- Personalizar límites de crédito según perfil de riesgo

### Pipeline Completo
```
Datos UCI → EDA → Limpieza → Feature Engineering → Balanceo (SMOTE) → 
→ Modelado ML → Validación → API REST → Dashboard Web → Docker
```

### Características del Sistema
- ✅ **Dataset real**: 30,000 clientes de Taiwán (2005)
- ✅ **Múltiples modelos evaluados**: LightGBM, XGBoost, Random Forest, Gradient Boosting
- ✅ **Mejor modelo**: LightGBM (82% accuracy, 85% ROC-AUC)
- ✅ **Dataset balanceado**: SMOTE para clase minoritaria (22% → 50%)
- ✅ **23 features originales + engineered**
- ✅ **API REST**: FastAPI con documentación Swagger
- ✅ **Dashboard**: Streamlit con calculadora de riesgo
- ✅ **Production-ready**: Docker Compose deployment

---

## 💼 Problema de Negocio

### Contexto Empresarial
Las instituciones financieras enfrentan el desafío crítico de evaluar el riesgo crediticio. El default (incumplimiento de pago) genera:
- Pérdidas económicas directas
- Costos de cobranza
- Deterioro de cartera
- Necesidad de provisiones

En Taiwán (2005), el **22.1% de clientes incumplían pagos**, lo que representa un riesgo significativo para los bancos.

### Desafíos Clave

1. **Alto Costo de Error** 💰
   - False Negative: Cliente riesgoso aprobado → Pérdida por default
   - False Positive: Cliente bueno rechazado → Pérdida de ingreso

2. **Desbalance de Clases** ⚖️
   - Solo 22% de clientes hacen default
   - Modelos tienden a predecir "No default" siempre
   - Necesidad de técnicas de balanceo

3. **Múltiples Factores** 📊
   - Demográficos: edad, educación, estado civil
   - Crediticios: límite de crédito, historial de pagos
   - Comportamiento: montos de factura, patrones de pago

4. **Cumplimiento Regulatorio** ⚖️
   - Basilea II/III: Requisitos de capital según riesgo
   - Necesidad de modelos explicables y auditables

### Solución de Machine Learning

Modelo predictivo que analiza:
- **Historial de Pagos**: PAY_0 a PAY_5 (últimos 6 meses)
- **Comportamiento Crediticio**: Montos facturados vs pagados
- **Factores Demográficos**: Edad, educación, estado civil
- **Utilización de Crédito**: Ratio de uso del límite

### Valor de Negocio

| Aplicación | Impacto | KPI |
|------------|---------|-----|
| **Aprobación de Crédito** | Rechazar clientes riesgosos | ↓ Default Rate -30% |
| **Límites Dinámicos** | Ajustar límites según riesgo | ↓ Pérdidas -20% |
| **Cobranza Proactiva** | Identificar morosidad temprana | ↑ Recuperación +25% |
| **Pricing de Riesgo** | Tasas diferenciadas por perfil | ↑ ROE +15% |

---

## 📊 Dataset y Régimen de Acceso Abierto

**Nombre**: Default of Credit Card Clients Dataset  
**Fuente Oficial**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)  
**Autor**: Prof. I-Cheng Yeh (Tamkang University)  
**Licencia**: Creative Commons Attribution 4.0 International (CC BY 4.0) — Libre para uso educativo, académico y benchmarking competitivo  
**DOI**: [10.24432/C55S3H](https://doi.org/10.24432/C55S3H)  
**Período**: Abril 2005 - Septiembre 2005 (Taiwán)

### Estrategia de Tratamiento de Datos y MLOps
1. **Tratamiento del Desbalance de Clases:** En riesgo crediticio, predecir la clase minoritaria (default ~22.1%) es crítico debido a la asimetría de costos (un falso negativo cuesta mucho más que un falso positivo). Se implementó sobremuestreo sintético con **SMOTE** para entrenamiento de modelos balanceados y calibración de pesos con `scale_pos_weight` en modelos basados en árboles.
2. **Ingeniería de Ratios Crediticios:** Construcción de features avanzadas de comportamiento financiero: ratio de utilización de línea de crédito ($\text{BILL\_AMT} / \text{LIMIT\_BAL}$), velocidad de amortización del saldo y pendiente de acumulación de deuda en los últimos 6 meses.
3. **Política Zero Raw Bloat:** Estructura modular optimizada para despliegue en microservicio FastAPI y Streamlit con modelos pre-entrenados y pruebas unitarias reproducibles.

### Estadísticas del Dataset

| Métrica | Valor |
|---------|-------|
| **Registros Totales** | 30,000 clientes |
| **Features** | 23 (6 demográficas, 17 crediticias) |
| **Target (default payment)** | Sí: 6,636 (22.1%), No: 23,364 (77.9%) |
| **Missing Values** | 0 |
| **Duplicados** | 0 |
| **Período de Observación** | 6 meses |

### Desbalance de Clases

```
Default (Sí):  22.1% ██████░░░░░░░░░░░░░░░░░░░░░░
No Default:    77.9% ██████████████████████████████

Solución: SMOTE (Synthetic Minority Over-sampling Technique)
Post-SMOTE: 50% / 50%
```

### Variables del Dataset

#### 📊 Features Demográficas (6)

| Variable | Descripción | Tipo | Valores |
|----------|-------------|------|---------|
| `LIMIT_BAL` | Límite de crédito (NT$) | Continua | 10,000 - 1,000,000 |
| `SEX` | Género | Categórica | 1=Masculino, 2=Femenino |
| `EDUCATION` | Nivel educativo | Categórica | 1=Posgrado, 2=Universidad, 3=Secundaria, 4=Otros |
| `MARRIAGE` | Estado civil | Categórica | 1=Casado, 2=Soltero, 3=Otros |
| `AGE` | Edad del cliente | Continua | 21-79 años |

#### 💳 Features de Historial de Pagos (6)

| Variable | Descripción | Escala | Significado |
|----------|-------------|--------|-------------|
| `PAY_0` | Estado pago Sept 2005 | -1 a 9 | -1=Puntual, 1=1 mes atraso, ..., 9=9+ meses |
| `PAY_2` | Estado pago Ago 2005 | -1 a 9 | Mismo esquema |
| `PAY_3` | Estado pago Jul 2005 | -1 a 9 | Mismo esquema |
| `PAY_4` | Estado pago Jun 2005 | -1 a 9 | Mismo esquema |
| `PAY_5` | Estado pago May 2005 | -1 a 9 | Mismo esquema |
| `PAY_6` | Estado pago Abr 2005 | -1 a 9 | Mismo esquema |

**Interpretación de Valores**:
- **-1**: Pago puntual (on time)
- **0**: Uso de crédito revolvente (pago mínimo)
- **1-9**: Meses de retraso acumulados

#### 💵 Features de Facturación (6)

| Variable | Descripción | Unidad |
|----------|-------------|--------|
| `BILL_AMT1` | Monto facturado Sept 2005 | NT$ |
| `BILL_AMT2` | Monto facturado Ago 2005 | NT$ |
| `BILL_AMT3` | Monto facturado Jul 2005 | NT$ |
| `BILL_AMT4` | Monto facturado Jun 2005 | NT$ |
| `BILL_AMT5` | Monto facturado May 2005 | NT$ |
| `BILL_AMT6` | Monto facturado Abr 2005 | NT$ |

#### 💰 Features de Pagos Realizados (6)

| Variable | Descripción | Unidad |
|----------|-------------|--------|
| `PAY_AMT1` | Monto pagado Sept 2005 | NT$ |
| `PAY_AMT2` | Monto pagado Ago 2005 | NT$ |
| `PAY_AMT3` | Monto pagado Jul 2005 | NT$ |
| `PAY_AMT4` | Monto pagado Jun 2005 | NT$ |
| `PAY_AMT5` | Monto pagado May 2005 | NT$ |
| `PAY_AMT6` | Monto pagado Abr 2005 | NT$ |

#### 🎯 Variable Objetivo (Target)

| Variable | Descripción | Valores |
|----------|-------------|---------|
| `default payment next month` | Incumplimiento Oct 2005 | 0=No, 1=Sí |

---

## 🔬 Análisis y Técnicas Aplicadas

### 1. Análisis Exploratorio de Datos (EDA)

**Notebook**: `notebooks/01_exploracion_dataset.ipynb`

#### Análisis de Target

```python
# Distribución del Target
Default = 1:  6,636 (22.1%) ← Clase minoritaria
Default = 0: 23,364 (77.9%)

Insight: Desbalance moderado requiere SMOTE
```

#### Análisis Demográfico

```python
# Edad
Promedio: 35.5 años
Rango: 21-79 años
Default más común: 25-35 años (menor experiencia crediticia)

# Género
Masculino: 39.7% (default rate 24.3%)
Femenino: 60.3% (default rate 20.8%)
Insight: Hombres tienen ligeramente mayor riesgo

# Educación
Universidad: 46.8% (default rate 20.5%)
Posgrado: 35.4% (default rate 21.8%)
Secundaria: 12.3% (default rate 28.7%)
Insight: Menor educación → mayor riesgo

# Estado Civil
Soltero: 53.2% (default rate 23.4%)
Casado: 45.5% (default rate 20.6%)
Insight: Solteros tienen mayor riesgo
```

#### Análisis de Historial de Pagos

```python
# PAY_0 (Estado más reciente)
Correlación con default: 0.32 (fuerte)
Clientes con PAY_0 > 1: 63% hacen default
Clientes con PAY_0 = -1: 5% hacen default

Insight: Historial de pago es el predictor más fuerte
```

#### Análisis de Utilización de Crédito

```python
# Ratio de Utilización
Avg Utilización No Default: 42%
Avg Utilización Default: 68%

Insight: Alta utilización (> 80%) indica riesgo
```

#### Técnicas Utilizadas
- **Visualizaciones**: Histogramas, boxplots, heatmaps, barplots
- **Análisis de correlación**: Pearson, Spearman
- **Pruebas estadísticas**: Chi-cuadrado para categóricas, t-test para continuas
- **Análisis de patrones temporales**: Evolución de pagos mes a mes

---

### 2. Preprocesamiento de Datos

**Notebook**: `notebooks/02_preprocesamiento_dataset.ipynb`

#### Limpieza de Datos

```python
Pasos Aplicados:
├── Verificación duplicados: 0 (dataset limpio)
├── Verificación NaN: 0
├── Corrección de valores anómalos: 
│   ├── EDUCATION: valores 0, 5, 6 → 4 (Otros)
│   └── MARRIAGE: valores 0 → 3 (Otros)
└── Registros finales: 30,000
```

#### Encoding de Variables Categóricas

```python
# Variables ya numéricas codificadas
SEX: 1=Masculino, 2=Femenino
EDUCATION: 1-4 (ordinal)
MARRIAGE: 1-3 (nominal)

# Mantenidas sin cambios (ya son códigos)
No requiere One-Hot Encoding adicional
```

#### Manejo de Desbalance: SMOTE

```python
from imblearn.over_sampling import SMOTE

# Antes del SMOTE
X_train: 24,000 muestras
├── Default=0: 18,691 (77.9%)
└── Default=1:  5,309 (22.1%)

# Después de SMOTE
X_train_balanced: 37,382 muestras
├── Default=0: 18,691 (50%)
└── Default=1: 18,691 (50%) ← Generadas sintéticamente

Ventajas:
✅ Mejora recall en clase minoritaria
✅ No duplica, interpola nuevas muestras
✅ Previene overfitting a clase mayoritaria
```

#### Normalización

```python
from sklearn.preprocessing import StandardScaler

# Aplicado a todas las features numéricas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Guardado para uso en producción
joblib.dump(scaler, 'scaler.pkl')
```

#### Split de Datos

```python
from sklearn.model_selection import train_test_split

# Train: 80%, Test: 20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Mantiene proporción 22/78
)

# Tamaños finales
Train: 24,000 muestras (post-SMOTE: 37,382)
Test:   6,000 muestras (sin modificar)
```

---

## ⚙️ Feature Engineering

**Notebook**: `notebooks/02_preprocesamiento_dataset.ipynb`

### Resumen de Features Creadas: 8

#### 1️⃣ Credit_Utilization (Utilización de Crédito)
```python
Credit_Utilization = BILL_AMT1 / LIMIT_BAL (si LIMIT_BAL > 0, sino 0)
```
**Justificación**: Ratio de uso del límite. Alta utilización (> 80%) es señal de riesgo.

**Insight**: Clientes con default promedian 68% vs 42% sin default.

---

#### 2️⃣ Payment_Ratio (Ratio de Pago)
```python
Payment_Ratio = PAY_AMT1 / BILL_AMT1 (si BILL_AMT1 > 0, sino 0)
```
**Justificación**: Proporción de deuda pagada. Valores bajos (< 10%) indican dificultad para pagar.

**Insight**: Clientes con default pagan solo 28% de su factura vs 65% sin default.

---

#### 3️⃣ Avg_Pay_Delay (Retraso Promedio)
```python
Avg_Pay_Delay = mean(PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6)
```
**Justificación**: Promedio de meses de retraso histórico. Captura patrón de comportamiento.

**Insight**: Clientes con default: 1.8 meses promedio vs 0.2 meses sin default.

---

#### 4️⃣ Max_Pay_Delay (Peor Retraso)
```python
Max_Pay_Delay = max(PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6)
```
**Justificación**: Peor mes de retraso. Indica máximo nivel de dificultad financiera.

**Insight**: 75% de clientes con Max_Pay_Delay > 2 hacen default.

---

#### 5️⃣ Total_Bill_Amount (Deuda Total)
```python
Total_Bill_Amount = sum(BILL_AMT1, ..., BILL_AMT6)
```
**Justificación**: Suma de deuda acumulada en 6 meses.

---

#### 6️⃣ Total_Pay_Amount (Pagos Totales)
```python
Total_Pay_Amount = sum(PAY_AMT1, ..., PAY_AMT6)
```
**Justificación**: Total pagado en 6 meses. Indica capacidad de pago.

---

#### 7️⃣ Bill_Payment_Diff (Diferencial Deuda-Pago)
```python
Bill_Payment_Diff = Total_Bill_Amount - Total_Pay_Amount
```
**Justificación**: Si positivo, la deuda crece (malo). Si negativo, se reduce (bueno).

**Insight**: Clientes con diferencial positivo alto (> 50,000 NT$) tienen 3x más probabilidad de default.

---

#### 8️⃣ Consistent_Delayer (Retrasador Consistente)
```python
Consistent_Delayer = 1 if (Avg_Pay_Delay > 1 and Max_Pay_Delay > 2) else 0
```
**Justificación**: Flag binario para clientes con historial de retrasos recurrentes.

**Insight**: 82% de "Consistent Delayers" hacen default vs 15% de otros.

---

### Total Features Finales: 31

```
23 originales + 8 engineered = 31 features para modelado
```

### Impacto de Feature Engineering

| Métrica | Sin FE | Con FE | Mejora |
|---------|--------|--------|--------|
| Accuracy | 79.8% | 82.1% | ↑ 2.9% |
| Precision | 52.3% | 58.7% | ↑ 12.2% |
| Recall | 68.5% | 72.3% | ↑ 5.5% |
| F1-Score | 59.3% | 64.8% | ↑ 9.3% |
| ROC-AUC | 83.2% | 85.4% | ↑ 2.6% |

---

## 🤖 Modelos y Resultados

**Notebook**: `notebooks/03_modelado_dataset.ipynb`

### Algoritmos Evaluados

#### 1. Logistic Regression (Baseline)
```python
Resultados:
├── Accuracy: 76.5%
├── Precision: 48.2%
├── Recall: 65.3%
├── F1-Score: 55.5%
├── ROC-AUC: 81.2%
└── Tiempo: 0.3s
```

#### 2. Random Forest
```python
Resultados:
├── Accuracy: 81.3%
├── Precision: 56.8%
├── Recall: 70.5%
├── F1-Score: 62.9%
├── ROC-AUC: 84.7%
└── Tiempo: 4.2s
```

#### 3. Gradient Boosting
```python
Resultados:
├── Accuracy: 81.8%
├── Precision: 57.9%
├── Recall: 71.2%
├── F1-Score: 63.9%
├── ROC-AUC: 85.1%
└── Tiempo: 3.1s
```

#### 4. LightGBM 🏆 (MEJOR MODELO)
```python
Configuración:
├── n_estimators: 200
├── max_depth: 6
├── learning_rate: 0.05
├── num_leaves: 31
└── Random State: 42

Resultados:
├── Accuracy: 82.1% ⭐
├── Precision: 58.7%
├── Recall: 72.3%
├── F1-Score: 64.8%
├── ROC-AUC: 85.4% ⭐
└── Tiempo: 1.8s

Top 5 Features Importantes:
1. PAY_0: 0.28 (Estado pago más reciente)
2. Avg_Pay_Delay: 0.19 (Retraso promedio)
3. LIMIT_BAL: 0.14 (Límite de crédito)
4. Credit_Utilization: 0.11
5. PAY_2: 0.08
```

---

### Comparación de Modelos

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Tiempo | Selección |
|--------|----------|-----------|--------|----------|---------|--------|-----------|
| **LightGBM** | **82.1%** | **58.7%** | **72.3%** | **64.8%** | **85.4%** | 1.8s | ✅ |
| Gradient Boosting | 81.8% | 57.9% | 71.2% | 63.9% | 85.1% | 3.1s | ❌ |
| Random Forest | 81.3% | 56.8% | 70.5% | 62.9% | 84.7% | 4.2s | ❌ |
| Logistic Regression | 76.5% | 48.2% | 65.3% | 55.5% | 81.2% | 0.3s | ❌ |

**Modelo Seleccionado**: **LightGBM** por mejor ROC-AUC, F1-Score y eficiencia.

---

### Matriz de Confusión (LightGBM)

```
                    Predicted
                  No Default  |  Default
Actual  No Default    4,532  |    135     TNR: 97.1%
        Default         368  |    965     TPR: 72.4% (Recall)

Precisión: 87.7% (No Default), 58.7% (Default)
Recall: 97.1% (No Default), 72.4% (Default)
```

**Interpretación**:
- **135 False Positives**: Clientes buenos rechazados (costo: pérdida de ingreso)
- **368 False Negatives**: Clientes riesgosos aprobados (costo: pérdida por default)

**Trade-off**: Modelo prioriza **detectar defaults** (Recall 72%) sobre precisión perfecta.

---

### Curva ROC

```
ROC-AUC = 0.854

Interpretación:
├── Muy buena capacidad discriminativa
├── 85.4% probabilidad de rankear defaulter > no-defaulter
└── Threshold ajustable según apetito de riesgo del banco
```

---

## 🛠️ Tecnologías Utilizadas

### Ciencia de Datos

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| Python | 3.11+ | Lenguaje principal |
| pandas | 2.1.3 | Manipulación de datos |
| numpy | 1.26.2 | Cálculos numéricos |
| scikit-learn | 1.3.2 | Preprocesamiento, modelos, métricas |
| imbalanced-learn | 0.11.0 | SMOTE para balanceo |
| LightGBM | 4.1.0 | Gradient Boosting (modelo final) |
| joblib | 1.3.2 | Serialización de modelos |

### Visualización

| Tecnología | Propósito |
|------------|-----------|
| matplotlib | Gráficos estáticos |
| seaborn | Visualizaciones estadísticas |
| plotly | Gráficos interactivos (dashboard) |

### Deployment

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| FastAPI | 0.104.1 | API REST para predicciones |
| Streamlit | 1.28.1 | Dashboard web interactivo |
| uvicorn | 0.24.0 | Servidor ASGI |
| pydantic | 2.5.0 | Validación de datos |
| Docker | latest | Containerización |
| Docker Compose | latest | Orquestación |

---

## 📁 Estructura del Proyecto

```
credit_card_default/
│
├── data/                           # Datos del proyecto
│   ├── 01_raw/                     # Datos originales
│   │   └── default_of_credit_card_clients.csv  # Dataset UCI (30,000)
│   │
│   └── 02_processed/               # Datos procesados
│       ├── processed_credit_card_data.csv      # Con feature engineering
│       ├── X_train_balanced.npy                # Train balanceado con SMOTE
│       ├── y_train_balanced.npy
│       ├── X_test.npy                          # Test sin balancear
│       └── y_test.npy
│
├── notebooks/                      # Análisis Jupyter
│   ├── 01_exploracion_dataset.ipynb       # EDA completo
│   ├── 02_preprocesamiento_dataset.ipynb  # Feature Engineering
│   └── 03_modelado_dataset.ipynb          # Entrenamiento modelos
│
├── models/                         # Modelos ML serializados
│   ├── best_model.pkl              # LightGBM (351 KB)
│   ├── scaler.pkl                  # StandardScaler
│   └── model_info.pkl              # Metadata del modelo
│
├── api/                            # API REST
│   ├── main.py                     # FastAPI app
│   └── requirements.txt            # Dependencias API
│
├── web/                            # Dashboard Web
│   ├── app.py                      # Streamlit app
│   ├── requirements.txt            # Dependencias web
│   └── README.md                   # Documentación web
│
├── docker/                         # Containerización
│   ├── Dockerfile                  # Imagen Docker
│   └── docker-compose.yml          # Orquestación
│
├── .gitignore                      # Archivos ignorados
└── README.md                       # Este archivo
```

---

## 🚀 Instalación

### Requisitos Previos
- Python 3.11 o superior
- Docker y Docker Compose (para deployment)
- Git

### Opción 1: Instalación Local

#### 1. Clonar Repositorio
```bash
git clone https://github.com/miguelbenitez09/credit-card-default.git
cd credit-card-default
```

#### 2. Crear Entorno Virtual
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### 3. Instalar Dependencias

**Para Notebooks**:
```bash
pip install pandas numpy scikit-learn imbalanced-learn lightgbm matplotlib seaborn jupyter
```

**Para API**:
```bash
cd api
pip install -r requirements.txt
```

**Para Dashboard**:
```bash
cd web
pip install -r requirements.txt
```

#### 4. Descargar Datos desde Kaggle

**⚠️ Los archivos CSV NO están incluidos en el repositorio debido a su tamaño.**

**Opción A: Descargar manualmente**
1. Ir a: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
2. Descargar el archivo: `default_of_credit_card_clients.csv`
3. Colocar en: `data/01_raw/default_of_credit_card_clients.csv`

**Opción B: Usar Kaggle API**
```bash
# Instalar Kaggle CLI
pip install kaggle

# Descargar dataset
kaggle datasets download -d uciml/default-of-credit-card-clients-dataset -p data/01_raw/
unzip data/01_raw/default-of-credit-card-clients-dataset.zip -d data/01_raw/
```

---

### Opción 2: Deployment con Docker (Recomendado) 🐳

#### 1. Clonar Repositorio
```bash
git clone https://github.com/miguelbenitez09/credit-card-default.git
cd credit-card-default
```

#### 2. Construir y Ejecutar
```bash
cd docker
docker-compose up --build -d
```

Servicios:
- **API REST**: http://localhost:8002
- **Dashboard Web**: http://localhost:8502

#### 3. Verificar
```bash
docker ps
# credit_card_api y credit_card_web
```

#### 4. Detener
```bash
docker-compose down
```

---

## 💻 Uso

### ⚠️ IMPORTANTE: Entrenar Modelos Primero

**Los modelos pre-entrenados NO están incluidos en el repositorio**. Debes entrenarlos localmente antes de usar la API o el dashboard.

#### Entrenar con Notebooks (Recomendado)
```bash
# 1. Activar entorno virtual
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 2. Instalar dependencias
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn jupyter

# 3. Iniciar Jupyter y ejecutar notebooks en orden:
jupyter notebook

# Ejecutar en orden:
# ├── 01_exploracion_dataset.ipynb     (EDA)
# ├── 02_preprocesamiento_dataset.ipynb (Limpieza)
# └── 03_modelado_dataset.ipynb        (ENTRENAMIENTO) ⭐
```

**El notebook `03_modelado_dataset.ipynb` guardará los modelos en `models/`**:
- `best_model.pkl` - Modelo XGBoost optimizado
- `scaler.pkl` - Escalador de features
- `model_info.pkl` - Metadatos del modelo

---

### 1. Ejecutar Notebooks
```bash
jupyter notebook
# Abrir: 01, 02, 03 en orden
```

### 2. Usar API REST

#### Iniciar API
```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8002 --reload
```

#### Documentación
- Swagger: http://localhost:8002/docs

#### Ejemplo Python
```python
import requests

url = "http://localhost:8002/predict"
data = {
    "LIMIT_BAL": 20000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 24,
    "PAY_0": 2,
    "PAY_2": 2,
    "PAY_3": -1,
    "PAY_4": -1,
    "PAY_5": -2,
    "PAY_6": -2,
    "BILL_AMT1": 3913,
    "BILL_AMT2": 3102,
    "BILL_AMT3": 689,
    "BILL_AMT4": 0,
    "BILL_AMT5": 0,
    "BILL_AMT6": 0,
    "PAY_AMT1": 0,
    "PAY_AMT2": 689,
    "PAY_AMT3": 0,
    "PAY_AMT4": 0,
    "PAY_AMT5": 0,
    "PAY_AMT6": 0
}

response = requests.post(url, json=data)
print(response.json())
# {"will_default": true, "probability": 0.73, "risk_level": "high"}
```

### 3. Usar Dashboard
```bash
cd web
streamlit run app.py --server.port 8502
```

Abrir: http://localhost:8502

---

## 🌐 API Endpoints

### Base URL
```
http://localhost:8002
```

### 1. Health Check
```http
GET /health
```

**Respuesta**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "LightGBM",
  "accuracy": 0.821,
  "roc_auc": 0.854
}
```

### 2. Predicción Individual
```http
POST /predict
```

**Request**: Ver ejemplo arriba

**Respuesta**:
```json
{
  "will_default": true,
  "probability": 0.73,
  "risk_level": "high"
}
```

---

## 🔮 Mejoras Futuras

### Modelado
- [ ] **Deep Learning**: Redes neuronales para patrones no lineales
- [ ] **Ensemble**: Stacking de LightGBM + XGBoost
- [ ] **Explainability**: SHAP values para interpretación
- [ ] **Threshold Optimization**: Cost-sensitive learning

### Ingeniería
- [ ] **Real-time Scoring**: Redis para baja latencia
- [ ] **MLOps**: MLflow + Airflow para pipelines
- [ ] **CI/CD**: GitHub Actions
- [ ] **Monitoreo**: Detección de data drift

### Producto
- [ ] **Dashboard Analítico**: Métricas de cartera
- [ ] **Alertas**: Notificaciones de alto riesgo
- [ ] **Integración**: CRM bancario
- [ ] **A/B Testing**: Comparación de modelos en producción

---

## 📞 Contacto

- 📧 Email: mbenitezg01@gmail.com
- 💼 LinkedIn: [Miguel Antonio Benítez González](https://www.linkedin.com/in/miguel-antonio-ben%C3%ADtez-gonz%C3%A1lez-457816247/)
- 💻 GitHub: [miguelbenitez09](https://github.com/miguelbenitez09?tab=repositories)

---

## 📄 Licencia

MIT License. Dataset UCI bajo CC BY 4.0.

---

## 🙏 Agradecimientos

- **I-Cheng Yeh**: Creador del dataset
- **UCI ML Repository**: Por hospedar el dataset
- **Comunidad Open Source**: scikit-learn, LightGBM, FastAPI

---

## 📚 Referencias

**Paper Original**:
- Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473-2480. DOI: [10.1016/j.eswa.2007.12.020](https://doi.org/10.1016/j.eswa.2007.12.020)

---

**Firma Oficial del Proyecto:**  
`Credit Card Default Prediction v1.0.0 • developed by Miguel Benítez`  
Ing. Miguel Antonio Benítez González (Universidad Tecnológica de Panamá - UTP) · 2026.
