###############################################################################
# SCRIPT COMPLETO: FLUJO VECM/VAR + PRUEBAS, DIAGNÓSTICOS Y GRÁFICOS (IRF)
###############################################################################

# LIBRERÍAS
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera

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
df['dummy_quiebre'] = (df.index >= '2021Q3').astype(int)

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

###############################################################################
# PASO 2: TEST ADF (niveles) con interpretación
###############################################################################
def adf_test_print(series, name=''):
    res = adfuller(series.dropna())
    stat, pval = res[0], res[1]
    print(f'\n--- ADF test: {name} ---')
    print(f'Estadístico ADF: {stat:.4f} | p-value: {pval:.4f}')
    if pval <= 0.05:
        print("✅ Check: Serie estacionaria (rechazamos raíz unitaria).")
    else:
        print("❌ No correcto: Serie no estacionaria (no rechazamos raíz unitaria).")

print("\n--- PASO 2: ADF en niveles ---")
for col in var_order:
    adf_test_print(data_levels[col], col)

# ADF en primeras diferencias (para confirmar I(1) cuando aplique)
print("\n--- ADF en primeras diferencias (comprobación I(1)) ---")
for col in var_order:
    adf_test_print(data_levels[col].diff().dropna(), col + "_diff")


###############################################################################
# PASO 3: TEST JOHANSEN (cointegración)
###############################################################################
print("\n=== TEST DE COINTEGRACIÓN JOHANSEN ===")
joh = coint_johansen(data_levels, det_order=0, k_ar_diff=4)  # k_ar_diff puede cambiar según selección de lag
print("Estadísticos TRACE:", joh.lr1.round(4))
print("Valores críticos (90,95,99%):")
print(joh.cvt)

num_coint = int((joh.lr1 > joh.cvt[:,1]).sum())
if num_coint >= 1:
    print(f"\n✔️ Check: Se detectan {num_coint} relaciones de cointegración (r = {num_coint}).")
else:
    print("\n❌ No correcto: No se detecta cointegración (r = 0).")

###############################################################################
# PASO 4: SELECCIÓN DE REZAGOS (VAR.select_order) Y ELECCIÓN final
###############################################################################
sel = VAR(data_levels).select_order(maxlags=5)
print("\n--- Selección de rezagos (VAR.select_order) ---")
print(sel.summary())

# Elige lag por AIC por defecto (puedes cambiar a BIC si prefieres)
lag_opt = sel.selected_orders['aic']
if np.isnan(lag_opt):
    # fallback si AIC no sugiere (casos raros)
    lag_opt = sel.selected_orders.get('bic', 4)
print("Lag óptimo (AIC):", lag_opt)

# Para VECM, k_ar_diff = lag_opt
k_ar_diff = int(lag_opt) if not np.isnan(lag_opt) else 4

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

###############################################################################
# PASO 6: DIAGNÓSTICOS COMPLETOS DEL VECM
###############################################################################
if num_coint >= 1:

    print("\n============================")
    print(" DIAGNÓSTICOS DEL MODELO VECM")
    print("============================")

    # Obtener residuos
    resid = vecm_res.resid
    print("\n--- 6.1: Residuos del VECM: forma ---")
    print(resid.shape)

    ###########################################################################
    # 6.2 AUTOCORRELACIÓN DE RESIDUOS (LJUNG–BOX POR VARIABLE)
    ###########################################################################
    print("\n--- 6.2: Test de autocorrelación (Ljung–Box) ---")
    lb_results = {}

    for i, col in enumerate(var_order):
        serie = resid[:, i]
        lb = acorr_ljungbox(serie, lags=[12], return_df=True)
        pval = lb['lb_pvalue'].values[0]
        lb_results[col] = pval
        print(f"{col}: p-value = {pval:.4f}  -> ",
              "Sin autocorrelación" if pval > 0.05 else "Autocorrelación detectada")

    ###########################################################################
    # 6.3 NORMALIDAD MULTIVARIANTE (Doornik–Hansen)
    ###########################################################################
    print("\n--- 6.3: Test de normalidad multivariante (Doornik–Hansen) ---")

    from numpy.linalg import inv
    from scipy.stats import chi2

    def doornik_hansen_test(X):
        n, k = X.shape
        Z = (X - X.mean(0)) @ np.linalg.cholesky(inv(np.cov(X, rowvar=False)))
        skew = (Z**3).sum(axis=0) / n
        kurt = (Z**4).sum(axis=0) / n - 3
        stat = n * (skew@skew + 0.25*(kurt@kurt))
        return stat, 2*k

    X = resid
    stat, df_dh = doornik_hansen_test(X)
    pval_dh = 1 - chi2.cdf(stat, df_dh)

    print(f"Doornik–Hansen χ²({df_dh}) = {stat:.4f} | p-value = {pval_dh:.4f}")
    print("Normalidad ACEPTADA" if pval_dh > 0.05 else "Normalidad RECHAZADA")

    ###########################################################################
    # 6.4 PRUEBA ARCH (heterocedasticidad)
    ###########################################################################
    print("\n--- 6.4: Test ARCH de heterocedasticidad ---")
    from statsmodels.stats.diagnostic import het_arch

    for i, col in enumerate(var_order):
        serie = resid[:, i]
        arch_test = het_arch(serie)
        p_arch = arch_test[1]
        print(f"{col}: p-value = {p_arch:.4f} -> ",
              "Homoscedástico" if p_arch > 0.05 else "Heterocedasticidad detectada")
    ###############################################################################
    # 6.5 ESTABILIDAD DEL VECM (VAR equivalente) — 100% compatible 0.14.5
    ###############################################################################

    print("\n--- 6.5: Estabilidad del VECM (VAR equivalente) — Compatible con 0.14.5 ---")

    try:
        alpha = vecm_res.alpha        # k × r
        beta = vecm_res.beta          # k × r

        # ==== Extraer gamma correctamente ====
        gamma_raw = vecm_res.gamma
        if gamma_raw is None:
            gamma_list = []
        else:
            gamma_arr = np.array(gamma_raw)
            if gamma_arr.ndim != 3:
                gamma_list = []
            else:
                gamma_list = [gamma_arr[i, :, :] for i in range(gamma_arr.shape[0])]

        k = alpha.shape[0]
        p = k_ar_diff

        # MATRIZ Π = αβ'
        Pi = alpha @ beta.T

        # ===== Construcción de matrices A1...Ap =====
        A_mats = []

        # A1 = I + Π + suma(Γ_i)
        A1 = np.eye(k) + Pi.copy()
        for g in gamma_list:
            if g.shape == (k, k):
                A1 += g
        A_mats.append(A1)

        # A2...Ap = -Γ_(i-1)
        for i in range(1, p):
            if i-1 < len(gamma_list):
                Ai = -gamma_list[i-1]
            else:
                Ai = np.zeros((k, k))
            A_mats.append(Ai)

        # ===== Matriz companion =====
        top = np.hstack(A_mats)

        if p > 1:
            bottom_left = np.eye(k*(p-1))
            bottom_right = np.zeros((k*(p-1), k))
            bottom = np.hstack([bottom_left, bottom_right])
            companion = np.vstack([top, bottom])
        else:
            companion = top

        # ===== Raíces =====
        roots = np.linalg.eigvals(companion)

        print("\nRaíces del VAR equivalente:")
        print(roots)
        print("¿Estable?:", "✔ SÍ" if np.all(np.abs(roots) < 1) else "❌ NO")

    except Exception as e:
        print("ERROR al construir la matriz VAR equivalente:", e)


"""
###############################################################################
# PASO 6: ESTIMAR VAR EN NIVELES (MISMO LAG) PARA IRF/FEVD Y DIAGNÓSTICOS
###############################################################################
print("\n--- Estimando VAR en niveles (mismo lag_opt para coherencia) ---")
var_model = VAR(data_levels)
var_res = var_model.fit(lag_opt)
print(var_res.summary())

# Comprobamos estabilidad (raíces del companion matrix)
roots = np.abs(var_res.roots)
print("\n--- Raíces del companion matrix (módulos) ---")
print(roots.round(4))
if (roots < 1).all():
    print("✅ Check: El sistema VAR es estable (todas las raíces están dentro del círculo unitario).")
else:
    print("❌ No correcto: El sistema VAR parece inestable (alguna raíz >= 1).")

###############################################################################
# PASO 7: DIAGNÓSTICOS DE RESIDUOS (Ljung-Box y Jarque-Bera)
###############################################################################
print("\n--- Diagnósticos de residuos ---")
resid = var_res.resid  # DataFrame (T x n)

# Ljung-Box (autocorrelación) para cada residual (lag 12)
print("\nLjung-Box (lag 12) p-values:")
for i, col in enumerate(var_order):
    lb = acorr_ljungbox(resid[col], lags=[12], return_df=True)
    pval = lb['lb_pvalue'].values[0]
    print(f"  - {col}: p-value = {pval:.4f} -> ", end='')
    if pval > 0.05:
        print("✅ Check: No autocorrelación significativa en residuos (aceptado).")
    else:
        print("❌ No correcto: Autocorrelación significativa (rechazado). Considera aumentar lags o modelar errores.")

# Jarque-Bera (normalidad)
print("\nJarque-Bera (normalidad) p-values:")
for i, col in enumerate(var_order):
    jb_stat, jb_p, _, _ = jarque_bera(resid[col])
    print(f"  - {col}: JB p-value = {jb_p:.4f} -> ", end='')
    if jb_p > 0.05:
        print("✅ Check: Residuales parecen normales.")
    else:
        print("❌ No correcto: Residuales NO normales (cola/asimetría).")

# Mostrar std de residuos
print("\nDesv std de residuos por variable:")
print(resid.std().round(6))

###############################################################################
# PASO 8: IMPULSE RESPONSE FUNCTIONS (IRF) + BOOTSTRAP CONFIDENCE INTERVALS
###############################################################################
# NOTA: el orden de las variables ya está en var_order y se usará para Cholesky
print("\n--- Impulse Response Functions (IRF) ---")
steps_irf = 20  # p.ej. 20 trimestres (5 años)
irf = var_res.irf(steps_irf)

# Ploteo general
fig = irf.plot(orth=True)
plt.suptitle('IRFs (Cholesky orthogonalized)')
plt.tight_layout()
plt.show()

# IRF por impulso-respuesta individual (ejemplos)
# Ejemplo: efecto de un shock a N_TIR sobre N_PBI
try:
    irf.plot(impulse='N_TIR', response='N_PBI', orth=True)
    plt.title('IRF: shock a TIR -> PBI')
    plt.show()
except Exception:
    pass

# Bootstrap para bandas de confianza (puede demorar; ajusta nrep)
print("Calculando bootstrap para bandas de confianza (nrep=500). Esto puede tardar.")
irf_bs = var_res.irf(steps_irf).boot(nrep=500)
irf_bs.plot(orth=True)
plt.suptitle('IRFs con bootstrap (intervalos)')
plt.tight_layout()
plt.show()

###############################################################################
# PASO 9: FEVD (Forecast Error Variance Decomposition)
###############################################################################
print("\n--- FEVD (descomposición de varianza de error de pronóstico) ---")
fevd = var_res.fevd(steps_irf)
# Mostrar resumen y plot
for step in [1, 4, 8, 20]:
    print(f"\nFEVD at horizon {step}:")
    df_fevd = pd.DataFrame(fevd.decomp[step-1], index=var_order, columns=var_order)
    # filas: variable que recibe la varianza; columnas: fuente del shock
    print(df_fevd.round(4))

fevd.plot()
plt.suptitle('FEVD')
plt.tight_layout()
plt.show()

###############################################################################
# PASO 10: OPCIONAL - SVAR simple (identificación alternativa)
# Si quieres, puedes descomentar la sección SVAR para estimar SVAR por A-matrix.
###############################################################################
# from statsmodels.tsa.api import SVAR
# print("\n--- Estimando SVAR tipo 'A' (opcional) ---")
# svar = SVAR(data_levels, svar_type='A', A=None, deterministic='n', order=lag_opt)
# svar_res = svar.fit()
# print(svar_res.summary())
# svar_irf = svar_res.irf(steps_irf)
# svar_irf.plot()
# plt.show()

###############################################################################
# GUARDAR GRÁFICOS (opcional) - EJEMPLO
###############################################################################
# plt.savefig('irf_plot.png', dpi=300)
# plt.savefig('fevd_plot.png', dpi=300)

print("\n--- FIN DEL FLUJO ---")
print("Orden de variables usado para identificación (Cholesky):", var_order)
print("Si quieres cambiar el orden para otra identificación estructural, dime cuál y actualizo el script.")
"""