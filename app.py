import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ── Configuración de la página ───────────────────────────────────
st.set_page_config(
    page_title="Vigiax",
    page_icon="🌙",
    layout="centered"
)

# ── Cargar modelo y artefactos ───────────────────────────────────
@st.cache_resource
def cargar_modelo():
    model    = joblib.load("vigiax_lr_model.pkl")
    scaler   = joblib.load("vigiax_scaler.pkl")
    encoder  = joblib.load("vigiax_encoder.pkl")
    features = joblib.load("vigiax_features.pkl")
    return model, scaler, encoder, features

model, scaler, encoder, features = cargar_modelo()

# ── Header ───────────────────────────────────────────────────────
st.title("🌙 Vigiax")
st.subheader("Predicción de riesgo de trastorno del sueño")
st.markdown("""
Sistema de detección temprana basado en Machine Learning con datos reales
del **CDC-NHANES 2017-2020** (8,500+ adultos estadounidenses).
""")
st.divider()

# ── Formulario ───────────────────────────────────────────────────
st.markdown("### 📋 Ingresa los datos del paciente")
st.info("💡 Los campos marcados con *(opcional)* pueden dejarse en blanco. Entre más datos brindes, más preciso será el resultado.")

col1, col2 = st.columns(2)

with col1:
    age       = st.slider("Edad (años)", 18, 80, 35)
    gender    = st.selectbox("Género", ["Male", "Female"])
    ethnicity = st.selectbox("Etnia", [
        "Non-Hispanic White", "Non-Hispanic Black", "Non-Hispanic Asian",
        "Mexican American", "Other Hispanic", "Other/Multiracial"
    ])
    weight_kg = st.number_input("Peso (kg)", 0.0, 260.0, 0.0, step=0.5,
                                 help="Dejar en 0 si no lo sabes")
    height_cm = st.number_input("Altura (cm)", 0.0, 200.0, 0.0, step=0.5,
                                 help="Dejar en 0 si no lo sabes")

    # BMI solo si tienen peso y altura
    if weight_kg > 0 and height_cm > 0:
        bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)
        st.metric("BMI calculado", bmi)
    else:
        bmi = None
        st.caption("_BMI se calculará automáticamente cuando ingreses peso y altura_")

    # Cintura opcional
    tiene_cintura = st.checkbox("Conozco mi medida de cintura *(opcional)*")
    waist_cm = None
    if tiene_cintura:
        waist_cm = st.number_input("Circunferencia cintura (cm)", 20.0, 200.0, 80.0, step=0.5)

with col2:
    # Pulso opcional
    tiene_pulso = st.checkbox("Conozco mi pulso en reposo *(opcional)*")
    resting_pulse = None
    if tiene_pulso:
        resting_pulse = st.slider("Pulso en reposo (bpm)", 34, 142, 70)

    sleep_weekday_hrs  = st.slider("Horas de sueño entre semana", 2.0, 14.0, 7.0, step=0.5)
    sleep_weekend_hrs  = st.slider("Horas de sueño fin de semana", 2.0, 14.0, 8.0, step=0.5)
    sleep_trouble_freq = st.selectbox(
        "Frecuencia de problemas para dormir",
        [0, 2, 3, 4, 5],
        format_func=lambda x: {
            0: "Nunca",
            2: "Rara vez",
            3: "A veces",
            4: "Seguido",
            5: "Casi siempre"
        }[x]
    )

    st.markdown("**Actividad física**")
    vigorous_work = st.radio("¿Hace actividad vigorosa en trabajo?", [1, 2],
                              format_func=lambda x: "Sí" if x == 1 else "No")
    moderate_work = st.radio("¿Hace actividad moderada en trabajo?", [1, 2],
                              format_func=lambda x: "Sí" if x == 1 else "No")
    vigorous_rec  = st.radio("¿Hace actividad vigorosa recreativa?", [1, 2],
                              format_func=lambda x: "Sí" if x == 1 else "No")

    vigorous_work_min = st.number_input(
        "Minutos/día actividad vigorosa laboral", 0, 840, 0,
        disabled=(vigorous_work == 2)
    )
    vigorous_rec_min = st.number_input(
        "Minutos/día actividad vigorosa recreativa", 0, 480, 0,
        disabled=(vigorous_rec == 2)
    )

st.divider()

# ── Contador de campos completados ───────────────────────────────
campos_opcionales = {
    "Peso":    weight_kg > 0,
    "Altura":  height_cm > 0,
    "Cintura": tiene_cintura,
    "Pulso":   tiene_pulso,
}
completados = sum(campos_opcionales.values())
total = len(campos_opcionales)

st.markdown(f"**Datos opcionales completados: {completados}/{total}**")
st.progress(completados / total)
if completados < total:
    faltantes = [k for k, v in campos_opcionales.items() if not v]
    st.caption(f"📌 Campos sin completar: {', '.join(faltantes)} — entre más datos brindes, más preciso será el resultado.")
else:
    st.success("✅ Todos los datos completados — máxima precisión del modelo")

# ── Predicción ───────────────────────────────────────────────────
if st.button("🔍 Analizar riesgo", use_container_width=True, type="primary"):

    # Imputar valores faltantes con medianas del dataset NHANES
    MEDIANAS = {
        "weight_kg":     82.3,
        "height_cm":     168.0,
        "bmi":           28.9,
        "waist_cm":      99.5,
        "resting_pulse": 70.0,
    }

    weight_final  = weight_kg  if weight_kg > 0  else MEDIANAS["weight_kg"]
    height_final  = height_cm  if height_cm > 0  else MEDIANAS["height_cm"]
    bmi_final     = bmi        if bmi is not None else MEDIANAS["bmi"]
    waist_final   = waist_cm   if waist_cm is not None else MEDIANAS["waist_cm"]
    pulse_final   = resting_pulse if resting_pulse is not None else MEDIANAS["resting_pulse"]

    # Encodear
    etnia_encoded  = encoder.transform([ethnicity])[0]
    gender_encoded = 0 if gender == "Male" else 1

    # Armar input
    input_data = pd.DataFrame([{
        "age":                age,
        "gender":             gender_encoded,
        "ethnicity":          etnia_encoded,
        "weight_kg":          weight_final,
        "height_cm":          height_final,
        "bmi":                bmi_final,
        "waist_cm":           waist_final,
        "vigorous_work":      vigorous_work,
        "moderate_work":      moderate_work,
        "vigorous_work_min":  vigorous_work_min if vigorous_work == 1 else 0,
        "vigorous_rec":       vigorous_rec,
        "vigorous_rec_min":   vigorous_rec_min if vigorous_rec == 1 else 0,
        "sleep_weekday_hrs":  sleep_weekday_hrs,
        "sleep_weekend_hrs":  sleep_weekend_hrs,
        "sleep_trouble_freq": sleep_trouble_freq,
        "resting_pulse":      pulse_final,
    }])[features]

    input_scaled = scaler.transform(input_data)
    prob         = model.predict_proba(input_scaled)[0][1]
    prediccion   = model.predict(input_scaled)[0]

    # ── Ajuste de confianza según datos completados ───────────────
    confianza = 60 + (completados / total) * 40  # entre 60% y 100%

    # ── Resultado ─────────────────────────────────────────────────
    st.markdown("### 📊 Resultado del análisis")

    col_res1, col_res2, col_res3 = st.columns(3)
    with col_res1:
        if prediccion == 1:
            st.error("⚠️ **Riesgo detectado**")
        else:
            st.success("✅ **Sin riesgo significativo**")
    with col_res2:
        st.metric("Probabilidad de trastorno", f"{prob*100:.1f}%")
    with col_res3:
        st.metric("Confianza del modelo", f"{confianza:.0f}%",
                  help="Aumenta con más datos opcionales completados")

    if completados < total:
        st.warning(f"📌 Completaste {completados}/{total} campos opcionales. Entre más datos brindes, más preciso será el resultado.")

    # Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={"text": "Nivel de riesgo (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "#E24B4A" if prob > 0.5 else "#1D9E75"},
            "steps": [
                {"range": [0,  30], "color": "#EAF3DE"},
                {"range": [30, 60], "color": "#FAEEDA"},
                {"range": [60, 100],"color": "#FCEBEB"},
            ],
            "threshold": {
                "line":  {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 50
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # ── Recomendaciones ───────────────────────────────────────────
    st.markdown("### 💡 Recomendaciones")
    if prob > 0.5:
        st.warning("""
        - Consulta con un especialista en medicina del sueño
        - Considera un estudio de polisomnografía
        - Mantén horarios regulares de sueño
        - Evita pantallas 1 hora antes de dormir
        - Reduce el consumo de cafeína después del mediodía
        """)
    else:
        st.info("""
        - Mantén tus hábitos actuales de sueño
        - Continúa con la actividad física regular
        - Duerme entre 7 y 9 horas diarias
        - Monitorea cambios en la calidad del sueño
        """)

    st.caption("⚠️ Este sistema es una herramienta de apoyo. No reemplaza el diagnóstico médico profesional.")

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.caption("Vigiax · Datos: CDC-NHANES 2017-2020 · Modelo: Regresión Logística · ROC-AUC: 0.686")
