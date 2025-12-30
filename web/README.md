# 💳 Interfaz Web para Predicción de Default

> **Interfaz web interactiva desarrollada con Streamlit para la API de predicción de default en tarjetas de crédito.**

---

## 👨‍💻 Autor

**Miguel Antonio Benítez González**
- 📧 Email: mbenitezg01@gmail.com
- 💻 GitHub: [https://github.com/miguelbenitez09](https://github.com/miguelbenitez09?tab=repositories)

---

## ✨ Características

- **Ingreso Manual**: Formulario para ingresar datos del cliente manualmente
- **Subida de JSON**: Cargar archivos JSON con datos de clientes
- **Verificación de Estado**: Comprobar si la API está funcionando
- **Predicciones Visuales**: Resultados de predicción con métricas y recomendaciones

## 📝 Requisitos

- Python 3.11+
- La API debe estar ejecutándose en:
  - Local: `http://localhost:8000`
  - Docker: `http://credit-default-api:8000` (interno) / `http://localhost:8002` (externo)

## 📦 Instalación

```bash
cd H_webInterface
pip install -r requirements.txt
```

## 🚀 Ejecución

### Ejecución Local

```bash
streamlit run app.py --server.port 8501
```

La interfaz estará disponible en `http://localhost:8501`

### Con Docker

Desde el directorio raíz del proyecto:

```bash
cd F_Docker
docker-compose up --build
```

La interfaz estará disponible en `http://localhost:8502`

## 🛠️ Uso

1. **Verificar que la API esté corriendo**: 
   - Local: `http://localhost:8000/health`
   - Docker: `http://localhost:8002/health`

2. **Ejecutar la interfaz web** (ver sección Ejecución)

3. **Usar las pestañas**:
   - **Ingreso Manual**: Formulario para ingresar datos del cliente
   - **Subir JSON**: Cargar archivos JSON con datos de clientes
   - **Resultados**: Visualizar predicciones y métricas

4. **Realizar predicciones** y visualizar resultados con recomendaciones de negocio

---

## 📝 Notas

- La interfaz detecta automáticamente si está corriendo en Docker o localmente
- Los puertos están configurados para no interferir con otros proyectos
- Asegúrate de que los modelos (`models/`) existan antes de ejecutar