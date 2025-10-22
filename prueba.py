################################################################################
# LIBRERÍAS
################################################################################
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

################################################################################
# PASO 0: CONFIGURACIÓN Y CARGA DE DATOS
################################################################################
path = Path('DATA/base2_tesis.xlsx')

# Cargar la hoja específica
df = pd.read_excel(path, sheet_name='base_tes')

# Limpiar nombres de columnas
df.columns = df.columns.str.strip().str.replace(' ', '_')

# --- Configurar índice temporal ---
if 'Año' in df.columns:
    # Si tus datos son trimestrales (Q = quarter)
    df['Año'] = pd.PeriodIndex(df['Año'], freq='Q')
    df.set_index('Año', inplace=True)

print("--- 1. Datos Cargados ---")
print(df.head(), "\n")

################################################################################
# RENOMBRAR VARIABLES
################################################################################
df = df.rename(columns={
    'IPC': 'N_IPC',
    'TCRM': 'N_TCRM',
    'PBI': 'N_PBI',
    'TIR': 'N_TIR',
    'S&P': 'N_S&P'
})

################################################################################
# LIMPIEZA DE DATOS
################################################################################
df_clean = df.dropna().copy()

print("--- 2. DataFrame Limpio (sin valores nulos) ---")
print(df_clean.head())
print("\nNúmero de observaciones:", len(df_clean))

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
# PRUEBA DE COINTEGRACIÓN DE JOHANSEN
################################################################################
print("\n--- 5. PRUEBA DE COINTEGRACIÓN DE JOHANSEN ---")
johansen_test = coint_johansen(df_clean, det_order=0, k_ar_diff=2)

trace_stat = johansen_test.lr1
crit_value = johansen_test.cvt
eigenvectors = johansen_test.evec
variables = df_n.columns.tolist()

print("Estadísticos de traza (95%):")
for i in range(len(trace_stat)):
    print(f"r = {i}: Estadístico traza = {trace_stat[i]:.4f} | Valor crítico = {crit_value[i,1]:.4f}")

num_coint = sum(trace_stat > crit_value[:,1])
print(f"\nRelaciones de cointegración detectadas: {num_coint}")

if num_coint > 0:
    print("\nVectores cointegrantes (β):")
    for i in range(num_coint):
        relation = " + ".join([f"{coef:.3f}*{var}" for coef, var in zip(eigenvectors[:, i], variables)])
        print(f"Relación {i+1}: {relation}")

################################################################################
# VISUALIZACIÓN DE COINTEGRACIÓN ENTRE DOS SERIES
################################################################################
def graficar_cointegracion(df, y_col, x_col):
    """Grafica relación de largo plazo y residuos."""
    y, x = df[y_col], df[x_col]
    model_lr = sm.OLS(y, sm.add_constant(x)).fit()
    y_eq = model_lr.predict(sm.add_constant(x))
    residuals = y - y_eq

    print(f"\nRelación {y_col} ~ {x_col}")
    print(model_lr.summary().tables[1])
    print(f"R² ajustado: {model_lr.rsquared_adj:.3f}")

    plt.figure(figsize=(12,6))
    plt.plot(df.index.astype(str), y, label=y_col, lw=2)
    plt.plot(df.index.astype(str), x, '--', label=x_col)
    plt.plot(df.index.astype(str), y_eq, ':', label='Equilibrio', color='green')
    plt.title(f'{y_col} vs {x_col} (Relación de largo plazo)')
    plt.xlabel('Periodo')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,5))
    plt.plot(df.index.astype(str), residuals, color='darkred')
    plt.axhline(0, color='black', ls='--')
    plt.title(f'Residuos: {y_col} ~ {x_col}')
    plt.xlabel('Periodo')
    plt.tight_layout()
    plt.show()

# Ejemplo:
graficar_cointegracion(df_n, 'N_S&P', 'N_TCRM')

################################################################################
# SELECCIÓN DE REZAGOS ÓPTIMOS
################################################################################
print("\n--- 6. SELECCIÓN DE REZAGOS ÓPTIMOS ---")
model_lag = VAR(df_clean[cols_n])
lag_selection = model_lag.select_order(maxlags=4)
print(lag_selection.summary())

best_lag = lag_selection.selected_orders.get('aic', 1)
print(f"\n✅ Rezagos óptimos según AIC: {best_lag}")

################################################################################
# ESTIMACIÓN DEL MODELO VECM
################################################################################
print("\n--- 7. ESTIMACIÓN DEL MODELO VECM ---")
vecm_model = VECM(
    df_clean[cols_n],
    k_ar_diff=best_lag,
    coint_rank=num_coint,
    deterministic='co'
)

vecm_fitted = vecm_model.fit()
print(vecm_fitted.summary())

################################################################################
# VALIDACIÓN DIAGNÓSTICA DEL MODELO VECM
################################################################################
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy import stats
import pandas as pd

print("\n==============================================")
print("🧪 VALIDACIÓN DE SUPUESTOS DEL MODELO VECM")
print("==============================================")

# Extraer residuales del modelo ajustado
resid = pd.DataFrame(vecm_fitted.resid, columns=df_clean[cols_n].columns)

################################################################################
# PRUEBA 1 - AUTOCORRELACIÓN SERIAL (LJUNG–BOX)
################################################################################
print("\n--- PRUEBA 1: AUTOCORRELACIÓN SERIAL (LJUNG–BOX) ---")

# Número de rezagos a evaluar (ajústalo según frecuencia de tus datos)
lags = [4]

for col in resid.columns:
    lb = acorr_ljungbox(resid[col], lags=lags, return_df=True)
    p_value = lb['lb_pvalue'].iloc[-1]

    if p_value < 0.05:
        print(f"❌ {col}: p-value = {p_value:.4f} → Hay autocorrelación en los residuos.")
    else:
        print(f"✅ {col}: p-value = {p_value:.4f} → No hay autocorrelación (residuos independientes).")

################################################################################
# PRUEBA 2 - HETEROCEDASTICIDAD (ARCH)
################################################################################
print("\n--- PRUEBA 2: HETEROCEDASTICIDAD (ARCH) ---")

for col in resid.columns:
    arch_test = het_arch(resid[col])
    f_stat, f_pvalue, lm_stat, lm_pvalue = arch_test

    print(f"\nResiduo de {col}:")
    print(f"Estadístico F: {f_stat:.4f}  |  p-valor: {f_pvalue:.4f}")
    print(f"Estadístico LM: {lm_stat:.4f} |  p-valor: {lm_pvalue:.4f}")

    if f_pvalue > 0.05 and lm_pvalue > 0.05:
        print(f"✅ No hay evidencia de heterocedasticidad en {col} (varianza constante).")
    else:
        print(f"⚠️ Se detecta heterocedasticidad en {col} (p < 0.05). Posible varianza no constante.")

################################################################################
# PRUEBA 3 - NORMALIDAD (Jarque–Bera)
################################################################################
print("\n--- PRUEBA 3: NORMALIDAD (Jarque–Bera) ---")

for col in resid.columns:
    jb_stat, jb_pvalue = stats.jarque_bera(resid[col])[:2]

    print(f"{col}: JB = {jb_stat:.3f}, p-valor = {jb_pvalue:.4f}")

    if jb_pvalue > 0.05:
        print(f"✅ No se rechaza la normalidad en {col} (residuos normales).")
    else:
        print(f"⚠️ Se rechaza la normalidad en {col} (p < 0.05).")

################################################################################
# PASO 7: PRUEBA DE ESTABILIDAD DEL MODELO VECM
################################################################################
import numpy as np
import matplotlib.pyplot as plt

print("\n--- 7. PRUEBA DE ESTABILIDAD DEL MODELO VECM ---")

# Extraer dimensiones
k_ar_diff = vecm_fitted.k_ar  # número de rezagos en diferencias
k_endog = vecm_fitted.neqs    # número de variables endógenas

# Construir matriz companion manualmente
A_matrices = vecm_fitted.coefs  # Lista de matrices A_i
companion_matrix = np.zeros((k_endog * k_ar_diff, k_endog * k_ar_diff))

# Bloques superiores (coeficientes del modelo)
for i in range(k_ar_diff):
    companion_matrix[:k_endog, i * k_endog:(i + 1) * k_endog] = A_matrices[i]

# Bloques inferiores (identidad)
if k_ar_diff > 1:
    companion_matrix[k_endog:, :-k_endog] = np.eye(k_endog * (k_ar_diff - 1))

# Calcular autovalores
eigvals = np.linalg.eigvals(companion_matrix)

# Mostrar resultados
print("\nAutovalores y sus módulos:")
for i, val in enumerate(eigvals, 1):
    print(f"λ{i} = {val.real:.6f} + {val.imag:.6f}i  |  |λ| = {abs(val):.6f}")

# Verificar estabilidad
if np.all(np.abs(eigvals) < 1):
    print("\n✅ El modelo VECM es ESTABLE (todas las raíces dentro del círculo unitario).")
else:
    print("\n⚠️ El modelo VECM es INESTABLE (existe al menos una raíz fuera del círculo unitario).")

# --- Gráfico de estabilidad ---
plt.figure(figsize=(6, 6))
plt.axhline(0, color='gray', linewidth=0.8)
plt.axvline(0, color='gray', linewidth=0.8)
circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--')
plt.gca().add_artist(circle)
plt.scatter(eigvals.real, eigvals.imag, color='royalblue', s=80)
plt.title('Prueba de Estabilidad del Modelo VECM')
plt.xlabel('Parte Real')
plt.ylabel('Parte Imaginaria')
plt.axis('equal')
plt.grid(alpha=0.3)
plt.show()
