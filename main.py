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