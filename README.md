# 🌙 Vigiax — Early Sleep Disorder Prediction System

> **Vigía** + **ax** · Un sistema que observa silenciosamente tu perfil de salud y emite una alerta temprana antes de que un trastorno del sueño sea formalmente diagnosticado.

---

## ¿Qué es Vigiax?

Vigiax es un modelo de clasificación basado en aprendizaje automático que predice la presencia de trastornos del sueño en adultos, utilizando exclusivamente variables de salud pasivas y de fácil recolección. No requiere estudios clínicos invasivos ni equipamiento especializado.

Fue desarrollado a partir de datos reales del **National Health and Nutrition Examination Survey (NHANES 2017–2020)** del Centers for Disease Control and Prevention (CDC) de los Estados Unidos.

---

## Datos

| Módulo NHANES | Contenido | Variables clave |
|---|---|---|
| `P_DEMO.XPT` | Demografía | Edad, género, etnia |
| `P_BMX.XPT` | Composición corporal | Peso, talla, IMC, cintura |
| `P_PAQ.XPT` | Actividad física | Ejercicio vigoroso/moderado, minutos |
| `P_BPXO.XPT` | Presión arterial | Frecuencia cardíaca en reposo |
| `P_SLQ.XPT` | Sueño | **Variable objetivo**, horas de sueño, problemas para dormir |

- **Variable objetivo:** `told_sleep_disorder` — diagnóstico clínico de trastorno del sueño reportado por un médico (1 = Sí, 0 = No)
- **Dataset final limpio:** ~5.400 registros · 14 variables predictoras

---

## Pipeline

```
Descarga XPT (CDC)
      ↓
Selección de columnas por módulo
      ↓
Integración por SEQN (ID único del participante)
      ↓
Limpieza:
  · Imputación con mediana (frecuencia cardíaca, cintura)
  · Eliminación de filas con peso/talla/IMC nulos (~2%)
  · Corrección lógica: minutos de ejercicio = 0 si reportó no hacer actividad
  · Codificación binaria: 1=Sí / 0=No
      ↓
Análisis exploratorio (heatmap de correlaciones)
      ↓
Entrenamiento y comparación de 5 modelos
      ↓
Exportación del modelo final con joblib
```

---

## Modelos evaluados

| Modelo | Notas |
|---|---|
| Regresión Logística | ✅ Modelo final — mejor balance interpretabilidad/desempeño |
| Random Forest | Evaluado con GridSearchCV |
| XGBoost | Evaluado con GridSearchCV |
| K-Nearest Neighbors | Evaluado con GridSearchCV |
| Árbol de Decisión | Evaluado con GridSearchCV |

**Hiperparámetros del modelo final:** `LogisticRegression(C=0.01, penalty='l2')`  
**Evaluación:** ROC-AUC · F1-score · Matriz de confusión · Curvas Precision-Recall

---

## Variables predictoras

```
age             → Edad del participante
gender          → Género (Male / Female)
ethnicity       → Etnia (6 categorías NHANES)
weight_kg       → Peso en kilogramos
height_cm       → Talla en centímetros
bmi             → Índice de Masa Corporal
waist_cm        → Circunferencia de cintura
resting_pulse   → Frecuencia cardíaca en reposo
vigorous_work   → Actividad vigorosa laboral (Sí/No)
moderate_work   → Actividad moderada laboral (Sí/No)
vigorous_rec    → Actividad vigorosa recreativa (Sí/No)
vigorous_work_min / vigorous_rec_min → Minutos de actividad vigorosa
sleep_weekend_hrs → Horas de sueño en fin de semana
sleep_trouble_freq → Frecuencia de problemas para dormir
```

---

## Archivos exportados

```
vigiax_lr_model.pkl   → Modelo de Regresión Logística entrenado
vigiax_scaler.pkl     → StandardScaler ajustado sobre los datos de entrenamiento
vigiax_encoder.pkl    → LabelEncoder para la variable objetivo
vigiax_features.pkl   → Lista de features en el orden esperado por el modelo
```

---

## Uso del modelo

```python
import joblib
import numpy as np

model   = joblib.load("vigiax_lr_model.pkl")
scaler  = joblib.load("vigiax_scaler.pkl")
encoder = joblib.load("vigiax_encoder.pkl")

# Ejemplo: adulto de 45 años, mujer, IMC 29.3, etc.
sample = np.array([[45, 1, 3, 72.0, 162.0, 29.3, 88.0, 74, 0, 1, 1, 0, 30, 7.0, 3]])
sample_scaled = scaler.transform(sample)
pred  = model.predict(sample_scaled)
proba = model.predict_proba(sample_scaled)

print("Diagnóstico predicho:", encoder.inverse_transform(pred)[0])
print("Probabilidad de trastorno:", round(proba[0][1], 3))
```

---

## Tecnologías

`Python` · `pandas` · `scikit-learn` · `XGBoost` · `matplotlib` · `seaborn` · `joblib` · `Google Colab`

---

## Equipo

| Nombre |
|---|
| John Fonseca |
| Saul Esteban Urrea Osorio |
| Samuel Daza Carvajal |
| Andres Felipe Gil Gallo |

**Fuente de datos:** [CDC NHANES 2017–2020](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2017-2020)  
**Entorno de desarrollo:** Google Colab
