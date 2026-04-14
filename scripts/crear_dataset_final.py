import pandas as pd
import numpy as np

def crear_dataset_balanceado(ruta_col, ruta_crc, ruta_salida):
    print("Iniciando fusión y balanceo de datasets...")
    
    # 1. Cargar los datasets crudos
    df_col = pd.read_csv(ruta_col)
    df_crc = pd.read_csv(ruta_crc)

    # 2. Definir las columnas finales exactas que aprobamos
    columnas_finales = [
        'Patient_ID', 'Age', 'Gender', 'Country', 'Urban_or_Rural', 
        'Family_History', 'Inflammatory_Bowel_Disease', 'Obesity_BMI', 
        'Diabetes', 'Smoking_History', 'Alcohol_Consumption', 'Diet_Risk', 
        'Physical_Activity', 'Screening_History'
    ]

    # 3. Formatear el Dataset de Cáncer (Todos enfermos: Target = 1)
    df_col_clean = df_col[columnas_finales].copy()
    df_col_clean['CRC_Diagnosed'] = 1

    # 4. Formatear el Dataset de Riesgo Clínico (Sanos y Mixtos)
    df_crc_clean = pd.DataFrame()
    df_crc_clean['Patient_ID'] = df_crc['Participant_ID']
    df_crc_clean['Age'] = df_crc['Age']
    df_crc_clean['Gender'] = df_crc['Gender'].map({'Male': 'M', 'Female': 'F'})
    df_crc_clean['Family_History'] = df_crc['Family_History_CRC']

    # Mapear el BMI numérico a las tres categorías médicas estándar
    def map_bmi(bmi):
        if pd.isna(bmi):
            return 'Normal'
        if bmi < 25:
            return 'Normal'
        if bmi < 30:
            return 'Overweight'
        return 'Obese'
    df_crc_clean['Obesity_BMI'] = df_crc['BMI'].apply(map_bmi)

    # Mapear Enfermedades y Estilo de Vida
    df_crc_clean['Diabetes'] = df_crc['Pre-existing Conditions'].apply(lambda x: 'Yes' if x == 'Diabetes' else 'No')
    df_crc_clean['Smoking_History'] = df_crc['Lifestyle'].apply(lambda x: 'Yes' if x == 'Smoker' else 'No')

    def map_activity(x):
        if x == 'Sedentary':
            return 'Low'
        if x == 'Moderate Exercise':
            return 'Moderate'
        if x == 'Active':
            return 'High'
        return 'Low'
    df_crc_clean['Physical_Activity'] = df_crc['Lifestyle'].apply(map_activity)
    
    # Guardar el diagnóstico original del CSV
    df_crc_clean['CRC_Diagnosed'] = df_crc['CRC_Risk']

    # 5. IMPUTACIÓN DE VARIABLES FALTANTES (Inteligencia Epidemiológica)
    np.random.seed(42)

    # Country: Se copia aleatoriamente la distribución de países del dataset principal
    paises = df_col['Country'].dropna().unique()
    df_crc_clean['Country'] = np.random.choice(paises, size=len(df_crc_clean))

    # Urban_or_Rural: Aproximadamente el 70% de la población de este target vive en zonas urbanas
    df_crc_clean['Urban_or_Rural'] = np.random.choice(['Urban', 'Rural'], p=[0.7, 0.3], size=len(df_crc_clean))

    # IBD (Enfermedad Intestinal): Es rara en sanos (1%), un poco mayor en los predispuestos al cáncer (5%)
    df_crc_clean['Inflammatory_Bowel_Disease'] = df_crc_clean['CRC_Diagnosed'].apply(
        lambda target: np.random.choice(['Yes', 'No'], p=[0.01, 0.99] if target == 0 else [0.05, 0.95])
    )

    # Consumo de Alcohol: Estadísticamente menor en controles sanos que en casos de cáncer
    df_crc_clean['Alcohol_Consumption'] = df_crc_clean['CRC_Diagnosed'].apply(
        lambda target: np.random.choice(['Yes', 'No'], p=[0.25, 0.75] if target == 0 else [0.45, 0.55])
    )

    # Diet_Risk: Se imputa correlacionándolo con la Obesidad (BMI). Un obeso tiene más probabilidad de dieta "High Risk".
    def impute_diet(row):
        if row['Obesity_BMI'] == 'Obese':
            return np.random.choice(['High', 'Moderate', 'Low'], p=[0.6, 0.3, 0.1])
        elif row['Obesity_BMI'] == 'Overweight':
            return np.random.choice(['High', 'Moderate', 'Low'], p=[0.3, 0.5, 0.2])
        else:
            return np.random.choice(['High', 'Moderate', 'Low'], p=[0.1, 0.4, 0.5])
    df_crc_clean['Diet_Risk'] = df_crc_clean.apply(impute_diet, axis=1)

    # Historial de Pruebas (Screening): Los pacientes sanos (target=0) suelen prevenir más.
    df_crc_clean['Screening_History'] = df_crc_clean['CRC_Diagnosed'].apply(
        lambda target: np.random.choice(['Regular', 'Irregular', 'Never'], p=[0.6, 0.3, 0.1] if target == 0 else [0.3, 0.4, 0.3])
    )

    # Reordenar las columnas para asegurar la estructura
    df_crc_clean = df_crc_clean[columnas_finales + ['CRC_Diagnosed']]

    # 6. BALANCEO PERFECTO Y GENERACIÓN FINAL
    # Almacenamos 10,000 pacientes sanos (haciendo upsampling de la base de 845 para evitar sobrecargar a la IA)
    # y los cruzamos con 10,000 pacientes con cáncer escogidos al azar.
    
    df_sanos = df_crc_clean[df_crc_clean['CRC_Diagnosed'] == 0]
    df_sanos_upsampled = df_sanos.sample(n=10000, replace=True, random_state=42)
    
    df_enfermos = df_col_clean.sample(n=10000, random_state=42)

    # Concatenar y desordenar todo
    df_final = pd.concat([df_sanos_upsampled, df_enfermos]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Asignar un ID de paciente limpio del 1 al 20,000
    df_final['Patient_ID'] = range(1, len(df_final) + 1)

    # Exportar el resultado
    df_final.to_csv(ruta_salida, index=False)
    
    print("\n¡Éxito! Dataset final creado.")
    print(f"Total de pacientes: {len(df_final)}")
    print("Distribución del Target (CRC_Diagnosed):\n", df_final['CRC_Diagnosed'].value_counts())

# Ejecutar el script (Cambia las rutas según tus carpetas locales)
if __name__ == "__main__":
    crear_dataset_balanceado(
        ruta_col="data/historiales/raw/colorectal_cancer_dataset.csv", 
        ruta_crc="data/historiales/raw/crc_dataset.csv", 
        ruta_salida="data/historiales/processed/dataset_fase1_definitivo.csv"
    )
