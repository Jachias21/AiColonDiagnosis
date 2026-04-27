import os
import json
import glob
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import shap

print("Iniciando Entrenamiento de Modelo Clínico: XGBoost (Cross-Validated)...")

# Crear carpeta de logs estilo CatBoost
xgb_info_dir = "xgboost_info"
os.makedirs(xgb_info_dir, exist_ok=True)
os.makedirs(os.path.join(xgb_info_dir, "test"), exist_ok=True)
os.makedirs(os.path.join(xgb_info_dir, "learn"), exist_ok=True)


# 1. CARGA Y PREPROCESAMIENTO
df = pd.read_csv("data/dataset_fase1_diagnostico.csv")
df = df.drop(columns=['Patient_ID'])

# Mapeos categóricos ordinales
mapeos = {
    'Obesity_BMI': {'Normal': 0, 'Overweight': 1, 'Obese': 2},
    'Diet_Risk': {'Low': 0, 'Moderate': 1, 'High': 2},
    'Physical_Activity': {'Low': 0, 'Moderate': 1, 'High': 2}, 
    'Family_History': {'No': 0, 'Yes': 1},
    'Inflammatory_Bowel_Disease': {'No': 0, 'Yes': 1},
    'Smoking_History': {'No': 0, 'Yes': 1},
    'Alcohol_Consumption': {'No': 0, 'Yes': 1},
    'Diabetes': {'No': 0, 'Yes': 1},
    'Screening_History': {'Never': 0, 'Irregular': 1, 'Regular': 2}
}

for col, mapping in mapeos.items():
    df[col] = df[col].map(mapping)

# One-Hot Encoding para las variables categóricas
cat_features_nombres = ['Gender', 'Country', 'Urban_or_Rural']
df = pd.get_dummies(df, columns=cat_features_nombres, dtype=int)

# Separar features y target
X = df.drop(columns=['CRC_Diagnosed'])
y = df['CRC_Diagnosed']

# Exportar las columnas generadas en el preprocesamiento para garantizar paridad predictiva
os.makedirs("models", exist_ok=True)
features_path = "models/xgboost_features.json"
with open(features_path, "w", encoding="utf-8") as f:
    json.dump(list(X.columns), f, indent=4)
print(f"✅ Columnas guardadas en: {features_path}")

# Calcular scale_pos_weight
negatives = (y == 0).sum()
positives = (y == 1).sum()
scale_pos_weight_value = negatives / positives

# 2. DEFINICIÓN DE RESTRICCIONES MONOTÓNICAS
# 1 = Aumenta riesgo, -1 = Reduce riesgo
monotone_constraints_dict = {
    'Age': 1, 
    'Family_History': 1, 
    'Inflammatory_Bowel_Disease': 1, 
    'Obesity_BMI': 1,
    'Smoking_History': 1,
    'Diabetes': 1,
    'Physical_Activity': -1,
    'Screening_History': -1
}

# Transformar dict a tupla del mismo orden que las columnas en X
constraints = []
for col in X.columns:
    if col in monotone_constraints_dict:
        constraints.append(monotone_constraints_dict[col])
    else:
        constraints.append(0)
monotone_constraints = tuple(constraints)

# 3. 5-FOLD CROSS VALIDATION
print("\nEjecutando 5-Fold Cross Validation para máxima robustez (Antifragilidad)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
models = []

params = {
    'n_estimators': 400,
    'learning_rate': 0.04,
    'max_depth': 5,
    'reg_lambda': 5.0,
    'eval_metric': 'auc',
    'monotone_constraints': monotone_constraints,
    'scale_pos_weight': scale_pos_weight_value,
    'random_state': 42,
    'n_jobs': -1
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    model = XGBClassifier(**params)
    model.fit(
        X_tr, y_tr, 
        eval_set=[(X_tr, y_tr), (X_va, y_va)], 
        verbose=False
    )
    
    # Extraer métricas del fold
    results = model.evals_result()
    
    # Guardarlas en estilo CatBoost para el mejor fold (o actualizando cada fold)
    if fold == skf.n_splits - 1:
        # Guardar en TSV para validation (test)
        test_df = pd.DataFrame({'iter': range(len(results['validation_1']['auc'])),
                                'auc': results['validation_1']['auc']})
        test_df.to_csv(os.path.join(xgb_info_dir, "test_error.tsv"), sep='\t', index=False)
        
        # Guardar en TSV para train (learn)
        learn_df = pd.DataFrame({'iter': range(len(results['validation_0']['auc'])),
                                 'auc': results['validation_0']['auc']})
        learn_df.to_csv(os.path.join(xgb_info_dir, "learn_error.tsv"), sep='\t', index=False)
        
        # Guardar JSON de training info (similar a catboost_training.json)
        training_info = {
            "meta": {
                "iteration_count": len(results['validation_0']['auc']),
                "name": "XGBoost experiment",
                "loss_function": "Logloss",
                "metrics": ["AUC"]
            },
            "iterations": [
                {
                    "iteration": i,
                    "learn": [results['validation_0']['auc'][i]],
                    "test": [results['validation_1']['auc'][i]]
                } for i in range(len(results['validation_0']['auc']))
            ]
        }
        with open(os.path.join(xgb_info_dir, "xgboost_training.json"), "w") as f:
            json.dump(training_info, f, indent=4)
            
        # Generar archivos binarios para TensorBoard igual que CatBoost
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer_learn = SummaryWriter(os.path.join(xgb_info_dir, "learn"))
            writer_test = SummaryWriter(os.path.join(xgb_info_dir, "test"))
            
            for i in range(len(results['validation_0']['auc'])):
                writer_learn.add_scalar("AUC", results['validation_0']['auc'][i], i)
                writer_test.add_scalar("AUC", results['validation_1']['auc'][i], i)
                
            writer_learn.close()
            writer_test.close()
            print(f"📈 Eventos TensorBoard generados automágicamente.")
        except ImportError:
            pass
            
        print(f"📦 Métricas de entrenamiento guardadas en '{xgb_info_dir}/'")
        
    oof_preds[val_idx] = model.predict_proba(X_va)[:, 1]
    models.append(model)
    print(f"Fold {fold+1}/5 completado.")

# 4. EVALUACIÓN Y REPORTING
print("\n--- Resultados Globales Consolidados ---")
oof_pred_class = (oof_preds >= 0.5).astype(int)

print("Classification Report Global:\n")
print(classification_report(y, oof_pred_class))

# 5. EXPLICABILIDAD SHAP
best_model = models[-1] # Seleccionamos el último fold
print("\nGenerando análisis de explicabilidad visual SHAP...")
explainer = shap.TreeExplainer(best_model)
X_sample = X.sample(n=min(2000, len(X)), random_state=42)
shap_values = explainer(X_sample)

os.makedirs("grafics_shap", exist_ok=True)
shap.summary_plot(shap_values, X_sample, show=False)
plt.title("Impacto Clínico de Variables (SHAP - XGBoost)")
plt.tight_layout()
plt.savefig("./grafics_shap/shap_summary_xgboost.png")
print("✅ Gráfico SHAP guardado como 'shap_summary_xgboost.png'.")

# 6. GUARDADO DEL MODELO
ruta_modelo = "./models/xgboost_crc_risk_model.json"
best_model.save_model(ruta_modelo)
print(f"✅ Modelo de Fase 1 validado y guardado exitosamente en: {ruta_modelo}")

# 7. GENERAR PREDICCIONES PARA PARIDAD
print("\nGenerando predicciones en red para pacientes de prueba...")
os.makedirs("resultados", exist_ok=True)
with open(features_path, "r", encoding="utf-8") as f:
    expected_columns = json.load(f)

for json_file in glob.glob("pacientes_prueba/*.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            patient_data = data[0]
        else:
            patient_data = data
            
    # Extraer columnas puras (mismas que en el entrenamiento)
    columnas_bases = [
        'Age', 'Gender', 'Country', 'Urban_or_Rural', 'Family_History', 
        'Inflammatory_Bowel_Disease', 'Obesity_BMI', 'Diabetes', 
        'Smoking_History', 'Alcohol_Consumption', 'Diet_Risk', 
        'Physical_Activity', 'Screening_History'
    ]
    datos_limpios = {col: patient_data.get(col, "Unknown") for col in columnas_bases}
    df_paciente = pd.DataFrame([datos_limpios])
    
    # Mapeo ordinal
    for col, mapping in mapeos.items():
        if col in df_paciente.columns:
            df_paciente[col] = df_paciente[col].map(mapping).fillna(0).astype(int)
            
    # Mapeo one-hot usando la misma estrategia
    df_paciente = pd.get_dummies(df_paciente, columns=cat_features_nombres, dtype=int)
    
    # Asegurar que todas las columnas esperadas estén presentes y en el orden correcto
    for col in expected_columns:
        if col not in df_paciente.columns:
            df_paciente[col] = 0
            
    df_paciente = df_paciente[expected_columns]
    
    # Inferencia con XGBoost
    proba = best_model.predict_proba(df_paciente)[0]
    risk = float(proba[1]) # Clase 1
    
    is_positive = risk >= 0.5
    
    identifier = patient_data.get("DNI", patient_data.get("Nombre", ""))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_id = "".join(c for c in str(identifier) if c.isalnum() or c in (' ', '_')).replace(' ', '_')
    filename = f"paciente_{safe_id}_{timestamp}.json"
    filepath = Path("resultados") / filename
    
    output_data = {
        "informacion_paciente": patient_data,
        "analisis_ia": {
            "fecha_analisis": datetime.now().isoformat(),
            "riesgo_detectado": bool(is_positive),
            "probabilidad_exacta": float(risk),
            "modelo_utilizado": "XGBoost"
        }
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

print(f"✅ Predicciones completadas y guardadas en resultados/ con formato XGBoost.")
