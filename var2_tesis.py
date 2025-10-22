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
df = pd.read_excel(path, sheet_name='base_tes', parse_dates=['Año'], index_col='Año')

df.index = pd.PeriodIndex(df.index, freq='Q')

# Limpiar nombres de columnas (buena práctica)
df.columns = df.columns.str.strip().str.replace(' ', '_')

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
# TEST DE ESTACIONARIEDAD (ADF)
################################################################################
def adf_test(series, name=''):
    """Test de Dickey-Fuller Aumentado."""
    result = adfuller(series.dropna())
    print(f'\n--- Test ADF: {name} ---')
    print(f'Estadístico ADF: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    if result[1] <= 0.05:
        print("✅ Serie estacionaria.")
    else:
        print("❌ Serie no estacionaria (tiene raíz unitaria).")

cols_n = [c for c in df_clean.columns if c.startswith('N_')]
df_n = df_clean[cols_n]

print("\n--- 3. Estacionariedad en series originales ---")
for name, col in df_n.items():
    adf_test(col, name)

# Diferenciadas
df_n_diff = df_n.diff().dropna()

print("\n--- 4. Estacionariedad en primeras diferencias ---")
for name, col in df_n_diff.items():
    adf_test(col, name + "_diff")

################################################################################
# PASO 4 : SELECCIÓN DEL ORDEN DE REZAGOS (LAGS) ÓPTIMO
################################################################################
from statsmodels.tsa.api import VAR

model = VAR(df_n_diff)
print("\n--- 4. Selección de Rezagos Óptimos (AIC, BIC, FPE, HQIC) ---")
lag_selection = model.select_order(maxlags=4)
print(lag_selection.summary())

optimal_lags = lag_selection.selected_orders['aic']
print(f"\n✅ Según el Criterio de Akaike (AIC), el número óptimo de rezagos es: {optimal_lags}")


################################################################################
# PASO 5: AJUSTE DEL MODELO VAR
################################################################################
model_fitted = model.fit(optimal_lags)

print("\n--- 6. Resumen del Modelo VAR ---")
print(model_fitted.summary())


"""
# Este modelo se estima solo para ver como se comportarian las variables con 8 rezagos
model = VAR(df[['ln_PBI', 'ln_SP', 'ln_TCRM']])
lag_order = model.select_order(maxlags=8)
print(lag_order.summary())
"""

################################################################################
# PRUEBA 1 - AUTOCORRELACION SERIAL DE LOS RESIDUOS (LGUN - BOX) 
################################################################################
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

# residuales en un DataFrame
resid = pd.DataFrame(model_fitted.resid, columns=model_fitted.names)

# test Ljung-Box para cada residuo (puedes cambiar lags)
lags = [optimal_lags]

print("\n=== PRUEBA DE AUTOCORRELACIÓN SERIAL (LJUNG–BOX) ===")
for col in resid:
    lb = acorr_ljungbox(resid[col], lags=lags, return_df=True)
    p_value = lb['lb_pvalue'].iloc[-1] 
# observa p-values; p < 0.05 indica autocorrelación
# Condicional para interpretar los resultados
    if p_value < 0.05:
        print(f"❌ {col}: p-value = {p_value:.4f} → Hay autocorrelación en los residuos.")
    else:
        print(f"✅ {col}: p-value = {p_value:.4f} → No hay autocorrelación (residuos independientes).")

################################################################################
# PRUEBA 2 - HETEROCEDASTICIDAD (ARCH)
################################################################################
from statsmodels.stats.diagnostic import het_arch

print("\n--- PRUEBA DE HETEROCEDASTICIDAD (ARCH) ---")
for col in resid:
    print(f"\nResiduo de {col}:")
    arch_test = het_arch(resid[col])
    f_stat, f_pvalue, lm_stat, lm_pvalue = arch_test

    print(f"Estadístico F: {f_stat:.4f}  |  p-valor: {f_pvalue:.4f}")
    print(f"Estadístico LM: {lm_stat:.4f} |  p-valor: {lm_pvalue:.4f}")

    # Condicional interpretativa
    if f_pvalue > 0.05 and lm_pvalue > 0.05:
        print(f"✅ No hay evidencia de heterocedasticidad en {col} (varianza constante).")
    else:
        print(f"⚠️ Se detecta heterocedasticidad en {col} (p < 0.05). Posible varianza no constante.")

################################################################################
# PRUEBA 3 - NORMALIDAD (Jarque-Bera)
################################################################################
from scipy import stats

print("\n--- PRUEBA DE NORMALIDAD (Jarque-Bera) ---")
for col in resid:
    jb = stats.jarque_bera(resid[col])
    jb_stat, jb_pvalue = jb.statistic, jb.pvalue

    print(f"{col}: JB = {jb_stat:.3f}, p-valor = {jb_pvalue:.4f}")

    # Condicional interpretativa
    if jb_pvalue > 0.05:
        print(f"✅ No se rechaza la normalidad para {col} (residuos normales).")
    else:
        print(f"⚠️ Se rechaza la normalidad para {col} (residuos no normales, p < 0.05).")

################################################################################
# PRUEBA 4 - ESTABILIDAD DEL MODELO VAR
################################################################################
print("\n--- PRUEBA DE ESTABILIDAD DEL MODELO VAR ---")

stable = model_fitted.is_stable()
print(f"¿El modelo es estable?: {'✅ Sí' if stable else '❌ No'}")

roots = model_fitted.roots
print("Raíces del polinomio AR:", np.round(roots, optimal_lags))

# Interpretación adicional
if np.all(np.abs(roots) < 1):
    print("✅ Todas las raíces están dentro del círculo unitario → el modelo es dinámicamente estable.")
else:
    print("⚠️ Algunas raíces están fuera del círculo unitario → el modelo es inestable.")

################################################################################
# PASO 6: FUNCIÓN IMPULSO-RESPUESTA (IRF) — GRAFICO MODERNO (corregido)
################################################################################
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Generar las funciones impulso-respuesta
irf = model_fitted.irf(8)  # horizonte de 8 períodos

# Configurar estilo visual moderno
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# Obtener nombres de variables y horizonte
variables = model_fitted.names
response = 'N_S&P'
h = irf.irfs.shape[0]

# Crear figura
fig, axes = plt.subplots(len(variables), 1, figsize=(8, 10), sharex=True)
fig.suptitle('Funciones Impulso-Respuesta para N_S&P', fontsize=18, weight='bold')

colors = sns.color_palette("deep", len(variables))

# Graficar cada IRF
for i, var in enumerate(variables):
    # Respuesta media
    irf_line = irf.irfs[:, variables.index(var), model_fitted.names.index(response)]
    # Error estándar y límites (±2σ)
    se = irf.stderr()[..., variables.index(var), model_fitted.names.index(response)]
    lower = irf_line - 2 * se
    upper = irf_line + 2 * se
    
    # Línea principal
    axes[i].plot(irf_line, color=colors[i], lw=2.2, label=f'Choque en {var}')
    # Banda de confianza
    axes[i].fill_between(np.arange(h), lower, upper, color=colors[i], alpha=0.2)
    
    axes[i].axhline(0, color='black', lw=1, linestyle='--')
    axes[i].set_ylabel('Respuesta')
    axes[i].legend(loc='upper right')
    axes[i].grid(True, linestyle='--', alpha=0.6)

axes[-1].set_xlabel('Horizonte (periodos)')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
