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
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.rcParams.update({'figure.max_open_warning': 0})
sns.set(style="whitegrid")

# AJUSTES
path = Path('DATA/base2_tesis.xlsx')   # <- ajusta si es necesario

###############################################################################
# PASO 0: CARGA Y LIMPIEZA
###############################################################################
df = pd.read_excel(path, sheet_name='base_tes', parse_dates=['Año'], index_col='Año')
df.index = pd.PeriodIndex(df.index, freq='Q')
df.columns = df.columns.str.strip().str.replace(' ', '_')

# RENOMBRAR
df = df.rename(columns={
    'IPC': 'N_IPC',
    'TCRM':'N_TCRM',
    'PBI':'N_PBI',
    'TIR':'N_TIR',
    'S&P':'N_S&P',
})

# DUMMY QUIEBRE
"""
df['dummy_quiebre'] = (df.index >= '2021Q3').astype(int)
"""

df['dummy_quiebre'] = ((df.index >= '2020Q1')& (df.index <= '2021Q4')).astype(int)

# LIMPIAR NA
df_clean = df.dropna().copy()

print("\n--- 1. Datos cargados y limpios ---")
print("Observaciones finales:", len(df_clean))
print("Columnas:", df_clean.columns.tolist())

# Variables que usaremos (en el orden elegido)
var_order = ['N_PBI', 'N_IPC', 'N_TIR', 'N_TCRM', 'N_S&P']
data_levels = df_clean[var_order].copy()

###############################################################################
# PASO 1: GRAFICOS EXPLORATORIOS
###############################################################################
# Series en niveles
fig, axes = plt.subplots(len(var_order), 1, figsize=(10, 2.5*len(var_order)), sharex=True)
for i, col in enumerate(var_order):
    axes[i].plot(data_levels.index.to_timestamp(), data_levels[col])
    axes[i].set_title(f'Serie en niveles: {col}')
    axes[i].axvline(pd.Timestamp('2021-07-01'), color='gray', linestyle='--', alpha=0.5)  # 2021Q3 referencia
plt.tight_layout()
plt.show()

# Series primeras diferencias (para visualizar)
data_diff = data_levels.diff().dropna()
fig, axes = plt.subplots(len(var_order), 1, figsize=(10, 2.5*len(var_order)), sharex=True)
for i, col in enumerate(var_order):
    axes[i].plot(data_diff.index.to_timestamp(), data_diff[col])
    axes[i].set_title(f'Primera diferencia: {col}')
plt.tight_layout()
plt.show()

# Matriz de correlación
plt.figure(figsize=(8,6))
sns.heatmap(data_levels.corr(), annot=True, fmt='.2f', cmap='vlag', center=0)
plt.title('Matriz de correlación (niveles)')
plt.show()

# ACF / PACF por variable (niveles)
for col in var_order:
    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(121)
    plot_acf(data_levels[col].dropna(), lags=20, ax=ax1, title=f'ACF {col}')
    ax2 = fig.add_subplot(122)
    plot_pacf(data_levels[col].dropna(), lags=20, ax=ax2, title=f'PACF {col}')
    plt.tight_layout()
    plt.show()

################################################################################
# PASO 1: TEST DE ESTACIONARIEDAD (ADF)
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
# PASO 2: TEST DE JOHANSEN (EN NIVELES)
################################################################################

from statsmodels.tsa.vector_ar.vecm import coint_johansen

print("\n=== TEST DE COINTEGRACIÓN JOHANSEN ===")

try:
    data_levels = df_clean[['N_IPC','N_TCRM','N_PBI','N_TIR','N_S&P']]
    joh = coint_johansen(data_levels, det_order=0, k_ar_diff=4)

    print("Estadísticos TRACE:")
    print(joh.lr1)
    print("\nValores críticos (90,95,99%):")
    print(joh.cvt)

    if (joh.lr1 > joh.cvt[:,1]).sum() >= 1:
        print("\n✔️ Se detecta al menos 1 relación de cointegración.")
        print("Interpretación: Podrías estimar un modelo VECM, aunque el VAR"
              " en diferencias sigue siendo válido si justificas tu enfoque.")
    else:
        print("\n✔️ No hay cointegración.")
        print("Interpretación: Es correcto estimar un VAR en diferencias sin VECM.")
except:
    print("❌ Error al ejecutar Johansen. Revisa que tus columnas existan.")
################################################################################
# PASO EXTRA: ESTIMACIÓN DEL MODELO VECM (opcional como robustez)
################################################################################

print("\n=== ESTIMACIÓN DEL MODELO VECM ===")

# Número de relaciones de cointegración detectadas
num_coint = (joh.lr1 > joh.cvt[:,1]).sum()

if num_coint >= 1:
    print(f"Se usará r = {num_coint} relaciones de cointegración.")

    # Construir el VECM
    vecm_model = VECM(
        data_levels,
        k_ar_diff=4,       # igual que Johansen → coherencia
        coint_rank=num_coint,
        deterministic="n"  # sin constante (igual que det_order=0)
    )

    vecm_res = vecm_model.fit()

    print("\n✔️ VECM estimado correctamente.")
    print(vecm_res.summary())

    # Parámetros de ajuste (velocidad de corrección)
    print("\n--- Parámetros de ajuste α ---")
    print(vecm_res.alpha)

    # Vectores cointegrantes (β)
    print("\n--- Vectores cointegrantes β ---")
    print(vecm_res.beta)
else:
    print("\nNo hay cointegración → No se estima VECM.")


################################################################################
# PASO 2: SELECCIÓN DEL ORDEN DE REZAGOS (LAGS) ÓPTIMO
################################################################################
from statsmodels.tsa.api import VAR

model = VAR(df_n_diff)
print("\n--- 4. Selección de Rezagos Óptimos (AIC, BIC, FPE, HQIC) ---")
lag_selection = model.select_order(maxlags=3)
print(lag_selection.summary())

optimal_lags = lag_selection.selected_orders['aic']
print(f"\n✅ Según el Criterio de Akaike (AIC), el número óptimo de rezagos es: {optimal_lags}")

###############################################################################
# PASO 3: ESTIMACIÓN DEL VAR CON DUMMY EXÓGENA
################################################################################
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.varmax import VARMAX

print("\n--- 6. Estimación del VAR con dummy exógena ---")

# Ajustar tamaño de dummy (dropna alinea las fechas)
df_dummy = df_clean.loc[df_n_diff.index, ['dummy_quiebre']]

# Estimación del VAR con dummy exógena
model = VAR(df_n_diff, exog=df_dummy)
model_fitted = model.fit(optimal_lags, trend='c')

print(model_fitted.summary())


###############################################################################
# PASO 4: PRUEBA DE CORRELACIÓN
################################################################################

# Seleccionar variables que quieres correlacionar
cols = df_n_diff.columns.tolist()

# Calcular matriz de correlaciones
corr_matrix = df_n_diff[cols].corr()

# Mostrar matriz en consola
print("\n=== Matriz de Correlaciones ===")
print(corr_matrix.round(3))

# Visualizar matriz con heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap de Correlaciones')
plt.show()
.0

################################################################################
# PASO 4.1: PRUEBA DE MULTICOLINEALIDAD (VIF)
################################################################################
print("\n=== 5. Prueba de Multicolinealidad (VIF) ===")

# Seleccionar solo variables endógenas (sin dummy)
X = df_n_diff.values
columnas = df_n_diff.columns

# Calcular VIF
vif_data = pd.DataFrame({
    'Variable': columnas,
    'VIF': [variance_inflation_factor(X, i) for i in range(X.shape[1])]
})

# Mostrar resultados con interpretación
for i in range(len(vif_data)):
    var = vif_data.loc[i, 'Variable']
    vif_val = vif_data.loc[i, 'VIF']

    if vif_val < 5:
        interpret = "✅ Bajo riesgo de multicolinealidad"
    elif vif_val < 10:
        interpret = "⚠️ Moderado riesgo de multicolinealidad"
    else:
        interpret = "❌ Alto riesgo de multicolinealidad"

    print(f"{var:>10}: VIF = {vif_val:.2f} → {interpret}")

# Mostrar tabla completa
print("\nTabla resumen del VIF:")
print(vif_data.round(3))

################################################################################
# PASO 5: PRUEBA DE CAUSALIDAD DE GRANGER
################################################################################
from statsmodels.tsa.stattools import grangercausalitytests

# --- Selecciona las variables endógenas de tu modelo VAR ---
# Asegúrate de que df y optimal_lags estén definidos antes de este punto
endog_vars = df_n_diff[['N_S&P', 'N_TIR','N_TCRM','N_IPC','N_PBI']]
max_lag = optimal_lags # Usa el número de rezagos óptimo de tu VAR
significance_level = 0.05 # Nivel de significancia

# Creamos un DataFrame para almacenar los resultados
results_df = pd.DataFrame(index=endog_vars.columns, columns=endog_vars.columns)

print("\n=== Procesando Prueba de Causalidad de Granger ===")

for caused in endog_vars.columns:
    for causing in endog_vars.columns:
        if caused != causing:
            # grangercausalitytests debe ejecutarse en el orden (y, x), donde x causa a y
            test_data = endog_vars[[caused, causing]]
            # verbose=False silencia las advertencias y el output intermedio
            test_result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)

            # Extrae el p-valor del test F para el número de rezagos óptimo
            # El p-valor está en test_result[lag][0]['ssr_ftest'][1]
            f_test_pvalue = test_result[max_lag][0]['ssr_ftest'][1]

            # Almacena el resultado en el DataFrame
            is_significant = f_test_pvalue < significance_level
            
            # Formateamos la celda como "p-valor (Resultado)"
            result_label = '✅ Causa' if is_significant else '❌ No Causa'
            results_df.loc[causing, caused] = f"{f_test_pvalue:.4f} ({result_label})"

# --- Imprimir la tabla de resultados ---
print("\n=== Matriz de Causalidad de Granger (H0: La Fila NO causa a la Columna) ===")
print("  Valor en celda: p-valor (Decisión)")
print("Columna causa a Fila") 
print("-" * 85)

# Transponemos el DataFrame para que la variable 'Causante' esté en las filas (más intuitivo)
# y la variable 'Causada' esté en las columnas.
# Rellenamos los diagonales con un guion
results_df = results_df.T.fillna('-')
print(results_df.to_markdown(numalign="left", stralign="left"))
print("-" * 85)

#La variable de la FILA (Causante) causa a la variable de la COLUMNA (Causada)

###############################################################################
# PASO 6: ESTABILIDAD DEL MODELO VAR CON DUMMY EXOGENA
###############################################################################
print("\n--- 8. Estabilidad del VAR ---")

try:
    # Raíces del polinomio AR
    ar_roots = model_fitted.roots
    print("Raíces del polinomio AR:")
    print(ar_roots)

    if all(np.abs(ar_roots) > 1):
        print("✅ El modelo es estable (todas las raíces están fuera del círculo unitario).")
    else:
        print("⚠️ El modelo NO es estable (algunas raíces están dentro del círculo unitario).")
except Exception as e:
    print("⚠️ Error al calcular las raíces AR:", str(e))

###############################################################################
# PASO 7: RESIDUOS DEL MODELO VAR CON DUMMY EXOGENA 
###############################################################################
print("\n--- 9. Análisis de los residuos ---")

# Extraer residuos en DataFrame con nombres correctos
resid = pd.DataFrame(model_fitted.resid, columns=model_fitted.names)
print(resid.head())

# Graficar residuos
fig, axes = plt.subplots(len(resid.columns), 1, figsize=(10, 6), sharex=True)
for i, col in enumerate(resid.columns):
    axes[i].plot(resid.index.to_timestamp(), resid[col], label=f"Residuos {col}")
    axes[i].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[i].legend()
plt.tight_layout()
plt.show()

################################################################################
# PRUEBA 7.1: AUTOCORRELACION SERIAL DE LOS RESIDUOS (LGUN - BOX) 
################################################################################
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

# residuales en un DataFrame
resid = pd.DataFrame(model_fitted.resid, columns=df_n_diff.columns)

# test Ljung-Box para cada residuo (puedes cambiar lags)
lags = [optimal_lags]

print("\n=== PRUEBA DE AUTOCORRELACIÓN SERIAL (LJUNG–BOX) ===")
for col in resid.columns:
    lb = acorr_ljungbox(resid[col], lags=lags, return_df=True)
    p_value = lb['lb_pvalue'].iloc[-1] 
# observa p-values; p < 0.05 indica autocorrelación
# Condicional para interpretar los resultados
    if p_value < 0.05:
        print(f"❌ {col}: p-value = {p_value:.4f} → Hay autocorrelación en los residuos.")
    else:
        print(f"✅ {col}: p-value = {p_value:.4f} → No hay autocorrelación (residuos independientes).")

################################################################################
# PRUEBA 7.2: PRUEBA DE VOLATILIDAD - HETEROCEDASTICIDAD (ARCH TEST)
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
# PRUEBA 7.3: NORMALIDAD (Jarque-Bera)
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
# PASO 8: FUNCIÓN IMPULSO-RESPUESTA (IRF) — GRAFICO MODERNO (corregido)
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

################################################################################
# PASO 9: DESCOMPOSICIÓN DE VARIANZA (FEVD) — NORMALIZADO
###############################################################################
print("\n=== DESCOMPOSICIÓN DE VARIANZA (FEVD) — NORMALIZADO ===")

try:
    fevd = model_fitted.fevd(8)

    # Revisar cuántos horizontes devolvió realmente
    fevd_h, n_vars, _ = fevd.decomp.shape

    print(f"\nEl FEVD ha generado {fevd_h} horizontes (0 a {fevd_h-1}).")
    print("Usamos solo los horizontes válidos.\n")

    variable_obj = 'N_S&P'
    idx = model_fitted.names.index(variable_obj)

    # Usamos solo horizontes reales
    horizontes_validos = list(range(fevd_h))

    for h in horizontes_validos:
        print(f"\n---------------- Horizonte {h} ----------------")

        # contribuciones crudas
        contrib = fevd.decomp[h, :, idx]

        # 🔥 NORMALIZACIÓN A 100%
        contrib_norm = contrib / contrib.sum() * 100

        for i, var in enumerate(model_fitted.names):
            pct = contrib_norm[i]    # YA ES PORCENTAJE (%)

            # Diagnóstico
            if pct < 0.05:
                diag = "Prácticamente sin efecto"
            elif pct < 5:
                diag = "Efecto débil"
            elif pct < 20:
                diag = "Contribución moderada"
            elif pct < 50:
                diag = "Alta contribución"
            else:
                diag = "Contribución dominante"

            print(f"{var:<10}: {pct:6.2f}% → {diag}")

        # Verificación
        total_sum = contrib_norm.sum()
        print(f"✔️ Suma total = {total_sum:.2f}% (Normalizada a 100%)")

    print("\n✔️ FEVD normalizada correctamente para todos los horizontes.\n")

except Exception as e:
    print("❌ Error en FEVD:", str(e))

###############################################################################
# GRAFICO DEL FEVD NORMALIZADO — VARIABLE OBJETIVO: N_S&P
###############################################################################

import matplotlib.pyplot as plt
import numpy as np

try:
    fevd = model_fitted.fevd(8)

    fevd_h, n_vars, _ = fevd.decomp.shape
    variable_obj = 'N_S&P'
    idx = model_fitted.names.index(variable_obj)

    horizontes = list(range(fevd_h))
    variables = model_fitted.names

    # Matriz para almacenar FEVD normalizado
    fevd_norm = np.zeros((fevd_h, n_vars))

    for h in horizontes:
        contrib = fevd.decomp[h, :, idx]
        fevd_norm[h, :] = contrib / contrib.sum() * 100   # Normalización

    # ============================
    # PLOT
    # ============================

    plt.figure(figsize=(10, 6))
    for j, shock in enumerate(variables):
        plt.plot(
            horizontes,
            fevd_norm[:, j],
            label=f"Shock en {shock}",
            linewidth=2
        )

    plt.title(f"Descomposición de Varianza (FEVD)\nVariable objetivo: {variable_obj}",
              fontsize=16, weight='bold')
    plt.xlabel("Horizonte", fontsize=12)
    plt.ylabel("Contribución (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print("❌ Error al graficar FEVD:", str(e))

################################################################################
# PASO 9: PRUEBA DE QUIEBRE ESTRUCTURAL (CHOW) — CORREGIDO
################################################################################

from scipy import stats
import statsmodels.api as sm

# 1️⃣ Preparar DataFrame para Chow Test
df_chow = df_n_diff.copy()
df_chow['dummy_quiebre'] = df_clean['dummy_quiebre'].loc[df_n_diff.index]
df_reset = df_chow.reset_index().rename(columns={'Año':'Trimestre'})

# 2️⃣ Función de Chow modificada
def chow_test_residual(df, split_index, dep_var, indep_vars, dummy_col='dummy_quiebre'):
    """
    Test de Chow ignorando el período donde la dummy es 1.
    - df: DataFrame con variables diferenciadas + dummy
    - split_index: posición donde se evalúa el quiebre
    - dep_var: variable dependiente
    - indep_vars: lista de independientes incluyendo dummy
    - dummy_col: nombre de la dummy
    """
    # Segmento 1: solo donde dummy=0
    mask1 = (df.index[:split_index] != df[dummy_col][:split_index]).any()
    Y1 = df.iloc[:split_index][dep_var][df[dummy_col][:split_index]==0]
    X1 = sm.add_constant(df.iloc[:split_index][indep_vars][df[dummy_col][:split_index]==0])

    # Segmento 2: solo donde dummy=0
    Y2 = df.iloc[split_index:][dep_var][df[dummy_col][split_index:]==0]
    X2 = sm.add_constant(df.iloc[split_index:][indep_vars][df[dummy_col][split_index:]==0])

    # Modelo completo fuera del rango de la dummy
    mask_full = df[dummy_col]==0
    Y_full = df[dep_var][mask_full]
    X_full = sm.add_constant(df[indep_vars][mask_full])

    # Ajustar OLS
    model1 = sm.OLS(Y1, X1).fit()
    model2 = sm.OLS(Y2, X2).fit()
    model_full = sm.OLS(Y_full, X_full).fit()

    # Estadístico F de Chow
    n1, n2 = len(Y1), len(Y2)
    k = X_full.shape[1]
    SSR1 = sum(model1.resid**2)
    SSR2 = sum(model2.resid**2)
    SSR_full = sum(model_full.resid**2)

    F = ((SSR_full - (SSR1 + SSR2)) / k) / ((SSR1 + SSR2) / (n1 + n2 - 2*k))
    p_value = 1 - stats.f.cdf(F, k, n1 + n2 - 2*k)
    return F, p_value

# 3️⃣ Evaluar puntos de quiebre fuera del rango de dummy
results = []
for i in range(8, len(df_reset) - 8):
    # ignorar trimestres dentro de la dummy
    if df_reset.loc[i, 'dummy_quiebre'] == 0:
        F, p = chow_test_residual(df_reset, i, dep_var='N_PBI', 
                                  indep_vars=["N_TIR","N_TCRM","N_IPC","N_S&P","dummy_quiebre"])
        results.append((df_reset.loc[i,'Trimestre'], F, p))

results_df = pd.DataFrame(results, columns=['Trimestre', 'F_stat', 'p_value'])

# 4️⃣ Mejor punto de quiebre residual
if not results_df.empty:
    best_break = results_df.loc[results_df['F_stat'].idxmax()]
    print("📊 Chow test (residual) — resultados fuera de dummy")
    print(results_df.to_string(index=False))
    print("\n🏆 Posible quiebre residual:")
    print(best_break)
    if best_break['p_value'] < 0.05:
        print(f"\n❌ Quiebre residual significativo en {best_break['Trimestre']}")
    else:
        print(f"\n✅ No se detectan quiebres estructurales residuales fuera de la dummy")
else:
    print("✅ Todos los trimestres dentro del rango de la dummy; no hay Chow residual que evaluar.")

