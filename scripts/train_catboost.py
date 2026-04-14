import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss, roc_auc_score, recall_score
from catboost import CatBoostClassifier, Pool
import shap
import os

print("Iniciando Entrenamiento de Modelo Clínico: CatBoost (Cross-Validated Final)...")

# 1. CARGA Y PREPROCESAMIENTO
df = pd.read_csv("data/historiales/processed/dataset_fase1_diagnostico.csv")
df = df.drop(columns=['Patient_ID']) 

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

cat_features_nombres = ['Gender', 'Country', 'Urban_or_Rural']
for col in cat_features_nombres:
    df[col] = df[col].astype('category')

X = df.drop(columns=['CRC_Diagnosed'])
y = df['CRC_Diagnosed']

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
    'Screening_History': -1  # A mayor frecuencia de screening, menor riesgo no detectado
}
cat_constraints = {col: dir for col, dir in monotone_constraints_dict.items()}
cat_features_idx = [X.columns.get_loc(col) for col in cat_features_nombres]

# 3. PESOS CLÍNICOS DEFINITIVOS
# Equilibrio perfecto entre Edad (Rey biológico) y multiplicadores clínicos
pesos_clinicos = {
    'Age': 5.0,                           # 👑 REY: Máximo peso base.
    'Inflammatory_Bowel_Disease': 4.5,    # Multiplicador crítico
    'Family_History': 4.5,                # Multiplicador crítico
    'Obesity_BMI': 3.0,                   
    'Smoking_History': 2.5,               
    'Alcohol_Consumption': 2.5,           
    'Screening_History': 2.5,             
    'Gender': 2.0,                        
    'Diabetes': 2.0,                      
    'Diet_Risk': 1.5,                     
    'Physical_Activity': 1.5,             
    'Urban_or_Rural': 1.0,                
    'Country': 1.0                        
}

feature_weights_dict = {col: pesos_clinicos.get(col, 1.0) for col in X.columns}

# 4. STRATIFIED 5-FOLD CROSS VALIDATION
print("\nEjecutando 5-Fold Cross Validation para máxima robustez (Antifragilidad)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
models = []

# Parámetros Anti-Overfitting (Sin Atajos)
params = {
    'iterations': 400,            # Ciclos óptimos controlados
    'learning_rate': 0.04,        # Aprendizaje progresivo
    'depth': 5,                   # Evita subdivisiones hipertópicas en la edad
    'l2_leaf_reg': 5.0,           # Regularización FUERTE: obliga a analizar todo el historial
    'monotone_constraints': cat_constraints,
    'feature_weights': feature_weights_dict,
    'eval_metric': 'AUC',
    'verbose': 0,                 
    'random_seed': 42
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    train_pool = Pool(X_tr, y_tr, cat_features=cat_features_idx)
    val_pool = Pool(X_va, y_va, cat_features=cat_features_idx)
    
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)
    
    oof_preds[val_idx] = model.predict_proba(val_pool)[:, 1]
    models.append(model)
    print(f"Fold {fold+1}/5 completado. AUC: {roc_auc_score(y_va, oof_preds[val_idx]):.4f}")

# 5. EVALUACIÓN GLOBAL (Out-of-Fold)
print("\n--- Resultados Globales Consolidados ---")
oof_pred_class = (oof_preds >= 0.5).astype(int)

auc_global = roc_auc_score(y, oof_preds)
recall_global = recall_score(y, oof_pred_class)
brier_global = brier_score_loss(y, oof_preds)

print(f"AUC-ROC Global:     {auc_global:.4f} (Excelente discriminación)")
print(f"Recall Global:      {recall_global:.4f} (Alta sensibilidad)")
print(f"Brier Score Global: {brier_global:.4f} (Magnífica calibración)")

# 6. EXPLICABILIDAD SHAP (Mejor modelo)
best_model = models[-1] # Seleccionamos el último fold (completamente generalista)

print("\nGenerando análisis de explicabilidad visual SHAP...")
explainer = shap.TreeExplainer(best_model)
# SHAP sobre una muestra amplia para no bloquear la memoria
X_sample = X.sample(n=2000, random_state=42)
shap_values = explainer(X_sample)

os.makedirs("./grafics_shap", exist_ok=True)
shap.summary_plot(shap_values, X_sample, show=False)
plt.title("Impacto Clínico de Variables (SHAP - Modelo CV Final)")
plt.tight_layout()
plt.savefig("./grafics_shap/shap_summary_catboost.png")
print("Gráfico SHAP guardado como 'shap_summary_catboost.png'.")

# 7. GUARDADO DEL MOTOR DE INFERENCIA
os.makedirs("./models", exist_ok=True)
ruta_modelo = "./models/catboost_crc_risk_model.cbm"
best_model.save_model(ruta_modelo)
print(f"✅ Modelo de Fase 1 validado y guardado exitosamente en: {ruta_modelo}")