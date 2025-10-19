import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
import statsmodels.api as sm

################################################################################
# PASO 0: CONFIGURACIÓN Y CARGA DE DATOS
################################################################################
# Ruta y carga del archivo Excel
path = Path('DATA/bd_tesis.xlsx')

# Cargar la hoja específica para la tesis
df = pd.read_excel(path, sheet_name='bdat_tes', index_col=None)

# Limpiar nombres de columnas (buena práctica)
df.columns = df.columns.str.strip().str.replace(' ', '_')

#  Si la columna 'Año' está como índice, traerla de vuelta
if 'Año' not in df.columns and df.index.name == 'Año':
    df.reset_index(inplace=True)

print("--- 1. Datos Cargados y Preparados ---")
print(df.head())
print("\nInformación del DataFrame:")
df.info()



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

################################################################################
# MODELO ALTERNATIVO: VECM (Vector Error Correction Model)
################################################################################

from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

# Suponemos que ya tienes df_log (series en logaritmos)
df_vecm = df_log.dropna()

################################################################################
# PASO 1: PRUEBA DE COINTEGRACIÓN DE JOHANSEN (con interpretación automática)
################################################################################
print("\n--- PRUEBA DE COINTEGRACIÓN DE JOHANSEN ---")

# Ejecutar prueba Johansen
johansen_test = coint_johansen(df_vecm, det_order=0, k_ar_diff=2)

# Resultados principales
trace_stat = johansen_test.lr1       # Estadístico de traza
crit_value = johansen_test.cvt       # Valores críticos (90%, 95%, 99%)
eigenvectors = johansen_test.evec    # Vectores cointegrantes
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
# VISUALIZACIÓN Y ANÁLISIS DE COINTEGRACIÓN ENTRE DOS SERIES
################################################################################
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

# --- Función general para analizar cointegración visual ---
def graficar_cointegracion(df, y_col, x_col):
    """
    Analiza y grafica la relación de cointegración entre dos series no estacionarias.
    Muestra tanto la trayectoria conjunta (largo plazo) como las desviaciones (residuos).
    """
    y = df[y_col]
    x = df[x_col]
    n_periods = len(df) # Número total de trimestres
    
    # Estimamos la relación de largo plazo (OLS)
    model_lr = sm.OLS(y, sm.add_constant(x)).fit()
    y_eq = model_lr.predict(sm.add_constant(x))
    residuals = y - y_eq

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
    # 🌟 CREACIÓN COMPACTA DE ETIQUETAS DE TIEMPO
    # Creamos un PeriodIndex de forma interna para generar las etiquetas.
    index_labels = pd.period_range(start='2016Q1', periods=n_periods, freq='Q').astype(str).tolist()
    index_positions = np.arange(n_periods)
    # ==================================================================

    # (Impresión de resultados omitida por brevedad, asumo que la mantienes)
    # print("\n===============================")
    # print(f"Relación de Cointegración: {y_col} ~ {x_col}")
    # ...

    # --- Gráfico 1: series cointegradas ---
    plt.figure(figsize=(12, 6))
    
    plt.plot(index_positions, y, label=y_col, lw=2)
    plt.plot(index_positions, x, '--', label=x_col)
    plt.plot(index_positions, y_eq, ':', label='Equilibrio', color='green')
    
    plt.title(f'{y_col} vs {x_col} (Relación de largo plazo)')
    plt.xlabel('Trimestre (Año)')
    plt.ylabel('Logaritmo natural')
    
    # Aplicación de las etiquetas
    plt.xticks(ticks=index_positions, labels=index_labels, rotation=45, ha='right')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Gráfico 2: residuos ---
    plt.figure(figsize=(12, 5))
    
    plt.plot(index_positions, residuals, color='darkred')
    
    plt.axhline(0, color='black', ls='--')
    plt.title(f'Residuos de {y_col} ~ {x_col}')
    plt.xlabel('Trimestre (Año)')
    plt.ylabel('Desviación del equilibrio')
    
    # Aplicación de las etiquetas
    plt.xticks(ticks=index_positions, labels=index_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()



# --- Ejecutar para un par de variables cointegradas ---
graficar_cointegracion(df_vecm, 'ln_SP', 'ln_PBI')



################################################################################
# PASO 2: PRUEBA DE SELECCIÓN DE REZAGOS ÓPTIMOS (para VECM / VAR)
################################################################################

from statsmodels.tsa.api import VAR

print("\n--- PRUEBA DE SELECCIÓN DE LAGS ÓPTIMOS ---")

# Usamos las series diferenciadas o logarítmicas según tu caso (df_log_diff o df_log)
# Para VECM se recomienda usar las series en nivel pero estacionarias en diferencia
model_lag = VAR(df_log.dropna())

# Evaluamos hasta 8 rezagos, por ejemplo
lag_selection = model_lag.select_order(maxlags=4)

# Mostramos la tabla con los valores de los criterios
print(lag_selection.summary())

# Extraemos el número de rezagos óptimos según cada criterio
optimal_lags = lag_selection.selected_orders
print("\nNúmero de rezagos óptimos según cada criterio:")
for criterio, valor in optimal_lags.items():
    print(f"{criterio.upper()}: {valor}")

# Interpretación automática (opcional)
best_lag = optimal_lags['aic']
print(f"\n✅ Según el Criterio de Akaike (AIC), el número óptimo de rezagos es: {best_lag}")

################################################################################
# PASO 3: Ajuste del modelo VECM
################################################################################
print("\n--- ESTIMACIÓN DEL MODELO VECM ---")

# Determinar número de cointegraciones (supongamos 1 si la traza > valor crítico)
vecm_model = VECM(df_vecm, k_ar_diff=best_lag, coint_rank=num_coint, deterministic='co')  # 'co' incluye constante en el término de cointegración
vecm_fitted = vecm_model.fit()

print(vecm_fitted.summary())

################################################################################
# PASO 4: Interpretación del término de corrección de error
################################################################################
print("\n--- INTERPRETACIÓN DEL TÉRMINO DE CORRECCIÓN DE ERROR ---")
print("Cada coeficiente alfa indica la velocidad de ajuste hacia el equilibrio de largo plazo.")
print("Signo negativo: la variable corrige desequilibrios; signo positivo: amplifica los choques.\n")

################################################################################
# PASO 5: Diagnóstico visual (opcional)
################################################################################
# Convertir los residuos a DataFrame con el mismo número de filas que vecm_fitted.resid
residuals = pd.DataFrame(
    vecm_fitted.resid,
    columns=df_vecm.columns,
    index=df_vecm.index[-vecm_fitted.resid.shape[0]:]  # ← ajusta automáticamente el índice
)

# Gráfico de residuos por variable
residuals.plot(subplots=True, figsize=(10, 6), title="Residuos del modelo VECM")
plt.tight_layout()
plt.show()





