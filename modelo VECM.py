import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import VECM

################################################################################
# PASO 0: CONFIGURACIÓN Y CARGA DE DATOS
################################################################################
# Ruta y carga del archivo Excel
path = Path('DATA/base2_tesis.xlsx')

# Cargar la hoja específica para la tesis
df = pd.read_excel(path, sheet_name='base_tes', index_col=None)

# Limpiar nombres de columnas (buena práctica)
df.columns = df.columns.str.strip().str.replace(' ', '_')

#  Si la columna 'Año' está como índice, traerla de vuelta
if 'Año' not in df.columns and df.index.name == 'Año':
    df.reset_index(inplace=True)

print("--- 1. Datos Cargados y Preparados ---")
print(df.head())
print("\nInformación del DataFrame:")
df.info()

# Renombrar columnas para trabajar más fácil
df = df.rename(columns={
    'IPC': 'N_IPC',
    'TCRM':'N_TCRM',
    'PBI':'N_PBI',
    'TIR':'N_TIR',
    'S&P':'N_S&P',
})

################################################################################
# PASO 3: TEST DE ESTACIONARIEDAD (DICKEY-FULLER AUMENTADO)
################################################################################

def adf_test(series, name=''):
    """Realiza el test de Dickey-Fuller Aumentado en una serie temporal."""
    result = adfuller(series.dropna())
    print(f'--- Test de Estacionariedad para: {name} ---')
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    if result[1] <= 0.05:
        print("Resultado: Evidencia fuerte contra la hipótesis nula (H0), la serie es estacionaria.\n")
    else:
        print("Resultado: Evidencia débil contra H0, la serie tiene una raíz unitaria y es no-estacionaria.\n")

# =====================================
# 3.1 Verificando Estacionariedad en las series originales
# =====================================
print("\n--- 3. Verificando Estacionariedad de las series originales ---")

# Filtrar columnas que comienzan con 'N_' (nombres normalizados)
cols_n = [col for col in df.columns if col.startswith('N_')]

# Crear DataFrame solo con las columnas originales
df_n = df[cols_n]

# Aplicar el test ADF a las series originales
for name, column in df_n.items():
    adf_test(column, name=name)

# =====================================
# 3.2 Verificando Estacionariedad en las series diferenciadas
# =====================================
print("\n--- 4. Verificando Estacionariedad de las series en primera diferencia ---")

# Diferenciar las series originales
df_n_diff = df_n.diff().dropna()

# Aplicar el test ADF a las series diferenciadas
for name, column in df_n_diff.items():
    adf_test(column, name=f'{name}_diff')






"""
################################################################################
# PASO 1: Convertirlo la serie en logaritmos 
################################################################################

df['ln_SP'] = np.log(df['S&P'])
df['ln_PBI'] = np.log(df['PBI'])
df['ln_TCRM'] = np.log(df['TCRM'])
df['ln_TIR'] = np.log(df['TIR'])  
df['ln_IPC'] = np.log(df['IPC'])
df['ln_Empleo'] = np.log(df['Empleo'])
    
# Creamos un nuevo DataFrame solo con las variables de interés (en logaritmos)
df_log = df[['ln_SP', 'ln_PBI', 'ln_TCRM', 'ln_TIR', 'ln_IPC', 'ln_Empleo']].copy()

print(df_log.head())
"""
################################################################################
# MODELO ALTERNATIVO: VECM (Vector Error Correction Model) - SERIES ORIGINALES
################################################################################

from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

# Seleccionamos las series originales (no logarítmicas)
cols_n = [col for col in df.columns if col.startswith('N_')]
df_vecm = df[cols_n].dropna()

################################################################################
# PASO 1: PRUEBA DE COINTEGRACIÓN DE JOHANSEN (con interpretación automática)
################################################################################
print("\n--- PRUEBA DE COINTEGRACIÓN DE JOHANSEN (Series Originales) ---")

# Ejecutar la prueba de cointegración de Johansen
# det_order = 0 → sin tendencia determinista
# k_ar_diff = 2 → número de rezagos en diferencias
johansen_test = coint_johansen(df_vecm, det_order=0, k_ar_diff=2)

# Resultados principales
trace_stat = johansen_test.lr1       # Estadístico de traza
crit_value = johansen_test.cvt       # Valores críticos (90%, 95%, 99%)
eigenvectors = johansen_test.evec    # Vectores cointegrantes (β)
variables = df_vecm.columns.tolist() # Nombres de las variables

# Mostrar resultados de la prueba
print("\nEstadísticos de traza y valores críticos (95%):")
for i in range(len(trace_stat)):
    print(f"r = {i}: Estadístico traza = {trace_stat[i]:.4f} | Valor crítico 95% = {crit_value[i,1]:.4f}")

# Evaluar número de relaciones de cointegración
num_coint = sum(trace_stat > crit_value[:,1])

if num_coint == 0:
    print("\n❌ No se encontró evidencia de cointegración entre las variables al nivel del 95%.")
else:
    print(f"\n✅ Se detectaron {num_coint} relaciones de cointegración (al 95% de confianza).")
    print("\nVectores cointegrantes estimados (relaciones de largo plazo):\n")

    for i in range(num_coint):
        print(f"Relación {i+1}: ", end="")
        relation = " + ".join([f"{coef:.3f}*{var}" for coef, var in zip(eigenvectors[:, i], variables)])
        print(relation)


