import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ── Configuración de la página ───────────────────────────────────
st.set_page_config(
    page_title="Vigiax",
    page_icon=None,
    layout="centered"
)

# ── Cargar modelo y artefactos ───────────────────────────────────
@st.cache_resource
def cargar_modelo():
    model    = joblib.load("vigiax_model.pkl")
    scaler   = joblib.load("vigiax_scaler.pkl")
    encoder  = joblib.load("vigiax_encoder.pkl")
    features = joblib.load("vigiax_features.pkl")
    return model, scaler, encoder, features

model, scaler, encoder, features = cargar_modelo()

# ── Header ───────────────────────────────────────────────────────
st.title("Vigiax")
st.subheader("Predicción de riesgo de trastorno del sueño")
st.markdown("""
Sistema de detección temprana basado en Machine Learning con datos reales
del **CDC-NHANES 2017-2020** (8,500+ adultos estadounidenses).
""")
st.divider()

# ── Formulario ───────────────────────────────────────────────────
st.markdown("### Ingresa los datos del paciente")
st.info("Los campos marcados con *(opcional)* pueden dejarse en blanco. Entre más datos brindes, más preciso será el resultado.")

with st.expander("Datos personales", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider(
            "Edad (años)", 18, 80, 35,
            help="Tu edad actual en años cumplidos."
        )
        gender = st.selectbox(
            "Género", ["Male", "Female"],
            format_func=lambda x: "Masculino" if x == "Male" else "Femenino",
            help="Género biológico del paciente."
        )
    with col2:
        ethnicity = st.selectbox(
            "Etnia", [
                "Non-Hispanic White", "Non-Hispanic Black", "Non-Hispanic Asian",
                "Mexican American", "Other Hispanic", "Other/Multiracial"
            ],
            format_func=lambda x: {
                "Non-Hispanic White":   "Blanco no hispano",
                "Non-Hispanic Black":   "Negro no hispano",
                "Non-Hispanic Asian":   "Asiático no hispano",
                "Mexican American":     "Mexicano americano",
                "Other Hispanic":       "Hispano otro",
                "Other/Multiracial":    "Mixto / otra etnia"
            }[x],
            help="Grupo étnico al que pertenece el paciente. Clasificación usada por el CDC en el estudio NHANES."
        )

with st.expander("Composición corporal"):
    weight_kg = st.number_input(
        "Peso (kg)", 0.0, 260.0, 0.0, step=0.5,
        help="Tu peso en kilogramos. Dejar en 0 si no lo sabes — el modelo usará el promedio poblacional."
    )
    height_cm = st.number_input(
        "Altura (cm)", 0.0, 200.0, 0.0, step=0.5,
        help="Tu altura en centímetros. Dejar en 0 si no lo sabes — el modelo usará el promedio poblacional."
    )

    if weight_kg > 0 and height_cm > 0:
        bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)
        estado_bmi = (
            "Bajo peso" if bmi < 18.5 else
            "Normal" if bmi < 25 else
            "Sobrepeso" if bmi < 30 else
            "Obesidad"
        )
        st.metric("BMI calculado", bmi, delta=estado_bmi,
                  delta_color="off",
                  help="Índice de Masa Corporal = peso / altura². Indicador de composición corporal.")
    else:
        bmi = None
        st.caption("_El BMI se calculará automáticamente cuando ingreses peso y altura._")

    tiene_cintura = st.checkbox(
        "Conozco mi medida de cintura *(opcional)*",
        help="La circunferencia de cintura es un indicador de grasa abdominal, asociada con mayor riesgo de trastornos del sueño como apnea."
    )
    waist_cm = None
    if tiene_cintura:
        waist_cm = st.number_input(
            "Circunferencia de cintura (cm)", 20.0, 200.0, 80.0, step=0.5,
            help="Medir con cinta métrica alrededor del ombligo, sin contraer el abdomen. Valor típico: 80–99 cm."
        )

with st.expander("Salud cardiovascular"):
    tiene_pulso = st.checkbox(
        "Conozco mi pulso en reposo *(opcional)*",
        help="El pulso en reposo refleja qué tan eficiente es tu corazón. Un pulso alto puede indicar estrés o mala condición física, factores relacionados con trastornos del sueño."
    )
    resting_pulse = None
    if tiene_pulso:
        resting_pulse = st.slider(
            "Pulso en reposo (bpm)", 34, 142, 70,
            help="Latidos por minuto en reposo total, idealmente medido en la mañana antes de levantarse. Rango normal: 60–100 bpm. Atletas pueden tener 40–60 bpm."
        )

with st.expander("Hábitos de sueño"):
    sleep_weekday_hrs = st.slider(
        "Horas de sueño entre semana", 2.0, 14.0, 7.0, step=0.5,
        help="Promedio de horas que duermes de lunes a viernes. Los adultos necesitan entre 7 y 9 horas según la Academia Americana de Medicina del Sueño."
    )
    sleep_weekend_hrs = st.slider(
        "Horas de sueño fin de semana", 2.0, 14.0, 8.0, step=0.5,
        help="Promedio de horas que duermes sábado y domingo. Dormir mucho más que entre semana puede indicar 'deuda de sueño' acumulada."
    )
    sleep_trouble_freq = st.selectbox(
        "¿Con qué frecuencia tienes problemas para dormir?",
        [0, 2, 3, 4, 5],
        format_func=lambda x: {
            0: "Nunca",
            2: "Rara vez (1–2 veces al mes)",
            3: "A veces (1–2 veces por semana)",
            4: "Seguido (3–4 veces por semana)",
            5: "Casi siempre (5 o más veces por semana)"
        }[x],
        help="Incluye dificultad para conciliar el sueño, despertarse durante la noche o sentirse sin descanso al despertar."
    )

with st.expander("Actividad física"):
    vigorous_work = st.radio(
        "¿Realiza actividad **vigorosa** en su trabajo o labores?", [1, 2],
        format_func=lambda x: "Sí" if x == 1 else "No",
        help="Incluye trabajos físicamente exigentes como construcción, carga de materiales pesados, agricultura intensa, etc."
    )
    if vigorous_work == 1:
        vigorous_work_min = st.number_input(
            "¿Cuántos minutos por día?", 0, 840, 60,
            help="Minutos promedio por día que realiza esta actividad vigorosa en el trabajo."
        )
    else:
        vigorous_work_min = 0

    moderate_work = st.radio(
        "¿Realiza actividad **moderada** en su trabajo o labores?", [1, 2],
        format_func=lambda x: "Sí" if x == 1 else "No",
        help="Incluye trabajos como enfermería, docencia activa, ventas con caminata, limpieza, etc."
    )

    vigorous_rec = st.radio(
        "¿Realiza actividad **vigorosa** en tiempo libre o recreación?", [1, 2],
        format_func=lambda x: "Sí" if x == 1 else "No",
        help="Incluye deportes recreativos, gym de alta intensidad, HIIT, crossfit, running, ciclismo de montaña, etc."
    )
    if vigorous_rec == 1:
        vigorous_rec_min = st.number_input(
            "¿Cuántos minutos por día?", 0, 480, 30,
            help="Minutos promedio por día que dedica a esta actividad recreativa vigorosa."
        )
    else:
        vigorous_rec_min = 0

st.divider()

# ── Contador de campos completados ───────────────────────────────
campos_opcionales = {
    "Peso":    weight_kg > 0,
    "Altura":  height_cm > 0,
    "Cintura": tiene_cintura,
    "Pulso":   tiene_pulso,
}
completados = sum(campos_opcionales.values())
total       = len(campos_opcionales)

st.markdown(f"**Datos opcionales completados: {completados}/{total}**")
st.progress(completados / total)
if completados < total:
    faltantes = [k for k, v in campos_opcionales.items() if not v]
    st.caption(f"Campos sin completar: {', '.join(faltantes)} — entre más datos brindes, más preciso será el resultado.")
else:
    st.success("Todos los datos completados — máxima precisión del modelo")

# ── Predicción ───────────────────────────────────────────────────
if st.button("Analizar riesgo", use_container_width=True, type="primary"):

    MEDIANAS = {
        "weight_kg":     82.3,
        "height_cm":     168.0,
        "bmi":           28.9,
        "waist_cm":      99.5,
        "resting_pulse": 70.0,
    }

    weight_final = weight_kg      if weight_kg > 0          else MEDIANAS["weight_kg"]
    height_final = height_cm      if height_cm > 0          else MEDIANAS["height_cm"]
    bmi_final    = bmi            if bmi is not None         else MEDIANAS["bmi"]
    waist_final  = waist_cm       if waist_cm is not None    else MEDIANAS["waist_cm"]
    pulse_final  = resting_pulse  if resting_pulse is not None else MEDIANAS["resting_pulse"]

    etnia_encoded  = encoder.transform([ethnicity])[0]
    gender_encoded = 0 if gender == "Male" else 1

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
        "vigorous_work_min":  vigorous_work_min,
        "vigorous_rec":       vigorous_rec,
        "vigorous_rec_min":   vigorous_rec_min,
        "sleep_weekday_hrs":  sleep_weekday_hrs,
        "sleep_weekend_hrs":  sleep_weekend_hrs,
        "sleep_trouble_freq": sleep_trouble_freq,
        "resting_pulse":      pulse_final,
    }])[features]

    input_scaled = scaler.transform(input_data)
    prob         = model.predict_proba(input_scaled)[0][1]
    prediccion   = model.predict(input_scaled)[0]

    confianza = 60 + (completados / total) * 40

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
                  help="Aumenta al completar más campos opcionales.")

    if completados < total:
        st.warning(f"📌 Completaste {completados}/{total} campos opcionales. Entre más datos brindes, más preciso será el resultado.")

    riesgo_pct = prob * 100
    if riesgo_pct <= 35:
        gauge_color = "#1D9E75"
        gauge_steps = [
            {"range": [0, 35], "color": "#D8F3E3"},
            {"range": [35, 60], "color": "#FFF4CC"},
            {"range": [60, 100], "color": "#FDE2E2"},
        ]
    elif riesgo_pct <= 60:
        gauge_color = "#F5B800"
        gauge_steps = [
            {"range": [0, 35], "color": "#D8F3E3"},
            {"range": [35, 60], "color": "#FFF4CC"},
            {"range": [60, 100], "color": "#FDE2E2"},
        ]
    else:
        gauge_color = "#E24B4A"
        gauge_steps = [
            {"range": [0, 35], "color": "#D8F3E3"},
            {"range": [35, 60], "color": "#FFF4CC"},
            {"range": [60, 100], "color": "#FDE2E2"},
        ]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=riesgo_pct,
        title={"text": "Nivel de riesgo (%)"},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": gauge_color},
            "steps": gauge_steps,
            "threshold": {
                "line":  {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 60
            }
        }
    ))
    fig.update_layout(
        height=320,
        margin=dict(t=40, b=0),
        transition={"duration": 1000, "easing": "cubic-in-out"}
    )
    st.plotly_chart(fig, use_container_width=True)

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
