"""
# main.py
import pandas as pd

# === 1. Cargar la base de datos ===
file_path = "DATA/expo-julio25.xlsx"  # cambia por la ruta real
df = pd.read_excel(file_path)
print(df)
"""
"""
# Verificar estructura
print(df.head())

# === 2. Limpieza básica ===
# Estandarizar nombres de columnas a minúsculas
df.columns = [c.strip().lower() for c in df.columns]

# Convertir columnas clave
df['año'] = df['año'].astype(int)
df['flujo'] = df['flujo'].str.upper()

# === 3. Tabla resumen total ===
resumen_total = df.groupby(['flujo', 'año']).size().reset_index(name='n_operaciones')

# Si tienes columna de valor FOB, podrías usar sum()
# resumen_total = df.groupby(['flujo', 'año'])['valor_fob'].sum().reset_index()

print("\n--- Resumen total ---")
print(resumen_total)

# === 4. Exportar a Excel para usar en el reporte ===
with pd.ExcelWriter("output/resumen_comercio.xlsx") as writer:
    resumen_total.to_excel(writer, sheet_name="Resumen", index=False)
"""

"""
import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as ts
import openpyxl
from tabulate import tabulate

excel_dataframe= openpyxl.load_workbook("expo-julio25.xlsx")

dataframe=excel_dataframe.active

data=[]

for row in range(1, dataframe.max_row): #recuento de los numeros de filas 
    print(row)
    """
"""
    _row = [row,]
    for col in dataframe.iter_cols(1,dataframe.max_column):
        _row.append(col[row].value)

    data.append(_row)
    
headers = ["S&P", "PBI", "TCRM", "TIR", "IPC", "Empleo"] 
headers_align = (("center",)*6)

print(tabulate(data,headers=headers,tablefmt="fancy_grid", colalign=headers_align))
"""


import pandas as pd
from tabulate import tabulate

# 1. Leer el Excel
df = pd.read_excel("DATA/expo-julio25.xlsx")

# 2. Estandarizar columnas
df.columns = [c.strip().lower() for c in df.columns]

# 3. Mostrar las primeras filas como tabla
print("\n📄 Primeras filas de la base:")
print(tabulate(df.head(10), headers=df.columns, tablefmt="fancy_grid", showindex=False))

# 4. Resumen simple por flujo y año
if 'flujo' in df.columns and 'año' in df.columns:
    resumen = df.groupby(['flujo', 'año']).size().reset_index(name='n_operaciones')
    print("\n📊 Resumen por flujo y año:")
    print(tabulate(resumen, headers=resumen.columns, tablefmt="fancy_grid", showindex=False))
else:
    print("⚠️ La base no tiene columnas 'flujo' y/o 'año'.")





















# esto va en modelo VECM


################################################################################
# VISUALIZACIÓN Y ANÁLISIS DE COINTEGRACIÓN ENTRE DOS SERIES
################################################################################
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

# --- Función general para analizar cointegración visual ---
def graficar_cointegracion(df, y_col, x_col):
    """
    #Analiza y grafica la relación de cointegración entre dos series no estacionarias.
    #Muestra tanto la trayectoria conjunta (largo plazo) como las desviaciones (residuos).
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
    """

"""
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

"""

"""






"""
################################################################################
# PASO 2: PRUEBA DE SELECCIÓN DE REZAGOS ÓPTIMOS (para VECM / VAR)
################################################################################

from statsmodels.tsa.api import VAR

print("\n--- PRUEBA DE SELECCIÓN DE LAGS ÓPTIMOS ---")

# Usamos las series diferenciadas o logarítmicas según tu caso (df_log_diff o df_log)
# Para VECM se recomienda usar las series en nivel pero estacionarias en diferencia
model_lag = VAR(df_vecm.dropna())

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
"""



##################################################################3
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.tools.sm_exceptions import MissingDataError


# --- Asumiendo que df_n_diff, optimal_lags, model_fitted están cargados ---
# (Variables endógenas, rezagos, y el objeto VARResults)

# Datos que me proporcionaste:
ddof = 22 # Número de parámetros en cada ecuación
T = 35    # Número de observaciones

# --------------------------------------------------------------------------
# PASO 1: Extracción de Residuos
# --------------------------------------------------------------------------
# La clave es que la longitud de los residuos debe ser igual a T.
# resid = pd.DataFrame(model_fitted.resid, columns=model_fitted.names) 
# Usamos un DataFrame de ejemplo para hacer el código ejecutable sin tu modelo completo:
# --- REEMPLAZA ESTO CON TUS DATOS REALES DE RESIDUOS ---
# Si tus residuos originales solo tienen 35 puntos, T=35 es correcto.
# Si estás ejecutando esto en el entorno de tu script anterior, resid ya está definido.
# k_vars = len(model_fitted.names)
# endog_names = model_fitted.names

# --- SIMULACIÓN DE RESIDUOS Y NOMBRES PARA REPRODUCIR LA LÓGICA ---
k_vars = 5
endog_names = ['N_IPC', 'N_TCRM', 'N_PBI', 'N_TIR', 'N_S&P']
# Generamos residuos aleatorios para que el código sea ejecutable:
resid_data = np.random.randn(T, k_vars) * 0.1 
resid = pd.DataFrame(resid_data, columns=endog_names)
resid.index = pd.PeriodIndex(pd.to_datetime(pd.date_range('2015Q1', periods=T, freq='Q')), freq='Q')
# -----------------------------------------------------------------

alpha = 0.05

print(f"Número de parámetros (ddof) en cada ecuación: {ddof}")
print(f"Número de observaciones (T): {T}")
print("\n--- TEST CUSUM (Estabilidad de Parámetros) RE-CORREGIDO ---")


# --------------------------------------------------------------------------
# Función para calcular el Valor Crítico de la prueba CUSUM (K-S)
# --------------------------------------------------------------------------
def get_cusum_crit_value_formal(T, ddof, alpha=0.05):
    """
    Calcula el valor crítico constante (límites K-S) para la comparación formal (Sup-B).
    Usa la aproximación de Brown, Durbin, Evans.
    """
    if T <= ddof:
        return np.inf # No se puede realizar el test
    
    # El valor formal (a) para K-S a 5% es 0.948.
    # El valor Sup-B se compara con este 'a'.
    a_alpha = 0.948
    return a_alpha

# --------------------------------------------------------------------------
# PASO 2: Aplicar y Graficar el Test CUSUM para cada Ecuación
# --------------------------------------------------------------------------

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(k_vars, 1, figsize=(10, 4 * k_vars), sharex=True)

if k_vars == 1:
    axes = [axes] 

cusum_results = []
crit_val_formal = get_cusum_crit_value_formal(T, ddof, alpha=alpha)


for i, col in enumerate(resid.columns):
    try:
        # APLICACIÓN CORRECTA: breaks_cusumolsresid solo devuelve 2 valores
        sup_b, cum_sum = breaks_cusumolsresid(resid[col], ddof=ddof)
        
        # --- Cálculo de Bandas Críticas para el Ploteo ---
        # La banda para el gráfico es una aproximación lineal: L = +/- a * sqrt((t-k)/(T-k)) * sqrt(T/T)
        # Usamos la aproximación más común (recta que une los límites K-S)
        
        # La prueba empieza en el punto 'ddof' (22 en este caso)
        T_eff = T - ddof
        t_index = np.arange(1, T_eff + 1)
        
        # Los límites del gráfico son una línea recta. 
        # C = a_alpha * sqrt(T-k)
        C_plot = 0.948 * np.sqrt(T_eff)
        
        # Pendiente (slope) de la línea de límites
        slope = C_plot / T_eff
        
        # Línea de límite: L(t) = C * (t / T_eff)
        # Se plotea sobre el índice de tiempo real a partir de 'ddof'
        upper_limit = slope * t_index
        lower_limit = -slope * t_index
        
        
        # --- Ploteo Manual ---
        
        # Convertir índice PeriodIndex a DatetimeIndex para el ploteo
        time_index = resid.index.to_timestamp()
        
        axes[i].plot(time_index, cum_sum, label='Suma Acumulada (CUSUM)', color='blue', lw=2)
        axes[i].axhline(0, color='black', lw=1, linestyle='--')
        
        # Graficar bandas críticas (solo a partir de 'ddof')
        axes[i].plot(time_index[ddof:], upper_limit, 
                     label=f'Límite {int((1-alpha)*100)}%', color='red', linestyle='--', lw=1.5)
        axes[i].plot(time_index[ddof:], lower_limit, 
                     color='red', linestyle='--', lw=1.5)
                     
        axes[i].set_title(f'Test CUSUM para la Ecuación: {col}')
        axes[i].legend(loc='upper left')

        # --- Interpretación Automatizada ---
        
        # La prueba formal CUSUM compara el estadístico Sup-B con el valor crítico 'a_alpha'
        # El estadístico Sup-B es el máximo valor absoluto de la suma acumulada ESCALADA.
        
        if sup_b > crit_val_formal:
            interpretacion = "❌ **RECHAZA H₀** (Inestabilidad): Parámetros no estables."
            resultado_formal = "Inestable"
        else:
            interpretacion = "✅ **NO RECHAZA H₀** (Estabilidad): Parámetros estables."
            resultado_formal = "Estable"

        # Imprimir el resultado interpretativo automáticamente
        print(f"\nEcuación {col}:")
        print(f"  Estadístico Sup-B: {sup_b:.4f} | Valor Crítico (K-S, 5%): {crit_val_formal:.4f}")
        print(f"  Resultado: {interpretacion}")
        
        cusum_results.append({
            'Ecuacion': col,
            'Sup_B': f'{sup_b:.4f}',
            'Critico_Formal': f'{crit_val_formal:.4f}',
            'Resultado': resultado_formal
        })

    except Exception as e:
        print(f"⚠️ Error al aplicar CUSUM a {col}: {e}")

axes[-1].set_xlabel('Periodo')
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------
# PASO 3: Mostrar Tabla Resumen
# --------------------------------------------------------------------------
print("\n--- RESUMEN DEL TEST CUSUM ---")
df_cusum = pd.DataFrame(cusum_results)
print(df_cusum.to_markdown(index=False, numalign="left", stralign="left"))

print("\n--- NOTA SOBRE LA INTERPRETACIÓN ---")
print(f"La Hipótesis Nula (H₀) es que los parámetros son estables. (Nivel de significancia = {int(alpha*100)}%)")
print("El Test CUSUM rechaza H₀ si el Estadístico Sup-B excede el Valor Crítico de Kolmogorov-Smirnov (0.948 para el 5%).")
