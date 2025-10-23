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
# Crear un nuevo DataFrame limpio (sin valores nulos)
################################################################################
df_clean = df.dropna().copy()

print("\n--- 2. DataFrame Limpio (sin valores nulos) ---")
print(df_clean.head())
print("\nNúmero de observaciones finales:", len(df_clean))

################################################################################
# Confirmar nombres finales de columnas
################################################################################
print("\nColumnas finales disponibles:")
print(df_clean.columns.tolist())

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

################################################################################
# MODELO ALTERNATIVO: VECM (Vector Error Correction Model) - SERIES ORIGINALES
################################################################################

from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

# Seleccionamos las series originales (no logarítmicas)
cols_n = [col for col in df.columns if col.startswith('N_')]
df_vecm = df[cols_n].dropna()
print(df_vecm.columns)

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

################################################################################
# VISUALIZACIÓN Y ANÁLISIS DE COINTEGRACIÓN ENTRE DOS SERIES (EN NIVEL)
################################################################################
def graficar_cointegracion(df, y_col, x_col):
    """
    Analiza y grafica la relación de cointegración entre dos series en nivel.
    Muestra tanto la trayectoria conjunta (largo plazo) como las desviaciones (residuos).
    """
    y = df[y_col]
    x = df[x_col]
    n_periods = len(df)  # Número total de observaciones (trimestres)

    # --- Estimamos la relación de largo plazo (OLS) ---
    model_lr = sm.OLS(y, sm.add_constant(x)).fit()
    y_eq = model_lr.predict(sm.add_constant(x))
    residuals = y - y_eq

    # --- Resultados principales ---
    print("\n===============================")
    print(f"Relación de Cointegración: {y_col} ~ {x_col}")
    print("===============================")
    print(model_lr.summary().tables[1])
    print(f"R² ajustado: {model_lr.rsquared_adj:.3f}")
    print(f"Media de los residuos: {residuals.mean():.6f}")
    print(f"Desviación estándar de residuos: {residuals.std():.6f}\n")

    # --- Interpretación automática ---
    if model_lr.rsquared_adj > 0.5:
        print("✅ Las series muestran una fuerte relación de largo plazo (cointegración probable).")
    elif model_lr.rsquared_adj > 0.2:
        print("⚠️ Relación moderada, posible cointegración parcial.")
    else:
        print("❌ Relación débil, poca evidencia de cointegración.\n")

    # ==================================================================
    # 🌟 CREACIÓN COMPACTA DE ETIQUETAS DE TIEMPO (si no tienes un índice temporal)
    index_labels = pd.period_range(start='2015Q2', periods=n_periods, freq='Q').astype(str).tolist()
    index_positions = np.arange(n_periods)
    # ==================================================================

    # --- Gráfico 1: series y equilibrio de largo plazo ---
    plt.figure(figsize=(12, 6))
    
    plt.plot(index_positions, y, label=y_col, lw=2)
    plt.plot(index_positions, x, '--', label=x_col)
    plt.plot(index_positions, y_eq, ':', label='Equilibrio estimado', color='green')
    
    plt.title(f'{y_col} vs {x_col} (Relación de largo plazo)')
    plt.xlabel('Trimestre (Año)')
    plt.ylabel('Nivel de las variables')
    
    plt.xticks(ticks=index_positions, labels=index_labels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Gráfico 2: residuos ---
    plt.figure(figsize=(12, 5))
    plt.plot(index_positions, residuals, color='darkred')
    plt.axhline(0, color='black', ls='--')
    plt.title(f'Residuos de la relación {y_col} ~ {x_col}')
    plt.xlabel('Trimestre (Año)')
    plt.ylabel('Desviación respecto al equilibrio')
    plt.xticks(ticks=index_positions, labels=index_labels, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# --- EJECUCIÓN EJEMPLO (ajusta los nombres a tus variables reales) ---
graficar_cointegracion(df_vecm, 'N_S&P', 'N_TCRM')

################################################################################
# PASO 2: PRUEBA DE SELECCIÓN DE REZAGOS ÓPTIMOS (para VECM / VAR)
################################################################################

from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM

print("\n--- 2. SELECCIÓN DE REZAGOS ÓPTIMOS ---")

# Usamos el DataFrame limpio y ya transformado (en niveles)
model_lag = VAR(df_clean.dropna())

# Evaluamos hasta 4 rezagos (puedes ajustar a 8 si tu serie es trimestral)
lag_selection = model_lag.select_order(maxlags=4)

# Mostramos la tabla con los criterios de información
print(lag_selection.summary())

# Extraemos el número de rezagos óptimos según cada criterio
optimal_lags = lag_selection.selected_orders
print("\nNúmero de rezagos óptimos según cada criterio:")
for criterio, valor in optimal_lags.items():
    print(f"{criterio.upper()}: {valor}")

# Selección principal (por AIC, pero podrías usar HQIC o BIC)
best_lag = optimal_lags.get('aic', 1)
print(f"\n✅ Según el Criterio de Akaike (AIC), el número óptimo de rezagos es: {best_lag}")

################################################################################
# PASO 3: ESTIMACIÓN DEL MODELO VECM
################################################################################
print("\n--- 3. ESTIMACIÓN DEL MODELO VECM ---")

# ⚠️ Asegúrate de definir antes `num_coint` según tu prueba de Johansen
# Ejemplo: num_coint = 1  # número de relaciones cointegradas detectadas

vecm_model = VECM(
    df_clean,
    k_ar_diff=best_lag,
    coint_rank=num_coint,
    deterministic='co'  # incluye constante en la relación de cointegración
)

vecm_fitted = vecm_model.fit()
print(vecm_fitted.summary())

