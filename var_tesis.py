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
path = Path('DATA/bd_tesis.xlsx')

# Cargar la hoja específica para la tesis
df = pd.read_excel(path, sheet_name='bdat_tes', parse_dates=['Fecha'], index_col='Fecha')

# Limpiar nombres de columnas (buena práctica)
df.columns = df.columns.str.strip().str.replace(' ', '_')

print("--- 1. Datos Cargados y Preparados ---")
print(df.head())
print("\nInformación del DataFrame:")
df.info()

"""
################################################################################
# PASO 1: VISUALIZACIÓN DE LAS SERIES
################################################################################
print("\n--- 2. Visualizando las series de tiempo ---")
fig, axes = plt.subplots(nrows=len(df.columns), ncols=1, dpi=120, figsize=(10, 12))
for i, ax in enumerate(axes.flatten()):
    data = df[df.columns[i]]
    ax.plot(data, color='blue', linewidth=1)
    ax.set_title(df.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)
plt.tight_layout()
plt.show()
"""

################################################################################
# PASO 2: Convertirlo la serie en logaritmos 
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
# 3.1 Verificando Estacionariedad SOLO en las series logarítmicas
# =====================================
print("\n--- 3. Verificando Estacionariedad de las series logarítmicas ---")

# Filtrar columnas que contienen 'ln_' en su nombre
cols_log = [col for col in df.columns if 'ln_' in col]

# Crear DataFrame solo con las columnas logarítmicas
df_log = df[cols_log]

# Aplicar el test ADF a las series en logaritmos
for name, column in df_log.items():
    adf_test(column, name=name)

# =====================================
# 3.2 Verificando Estacionariedad en las series logarítmicas diferenciadas
# =====================================
print("\n--- 4. Verificando Estacionariedad de las series logarítmicas en primera diferencia ---")

# Diferenciar solo las columnas logarítmicas
df_log_diff = df_log.diff().dropna()

# Aplicar el test ADF a las series logarítmicas diferenciadas
for name, column in df_log_diff.items():
    adf_test(column, name=f'{name}_diff')


################################################################################
# PASO 4 : SELECCIÓN DEL ORDEN DE REZAGOS (LAGS) ÓPTIMO
################################################################################
from statsmodels.tsa.api import VAR

model = VAR(df_log_diff)
print("\n--- 4. Selección de Rezagos Óptimos (AIC, BIC, FPE, HQIC) ---")
lag_selection = model.select_order(maxlags=4)
print(lag_selection.summary())

optimal_lags = lag_selection.selected_orders['aic']
print(f"Rezagos óptimos según AIC: {optimal_lags}")

"""
# Usaremos el valor sugerido por el AIC, que es común en la práctica
optimal_lags = lag_selection.aic
print(f"\nRezagos óptimos seleccionados (AIC): {optimal_lags}")

"""

################################################################################
# PASO 5: AJUSTE DEL MODELO VAR
################################################################################
model_fitted = model.fit(optimal_lags)

print("\n--- 6. Resumen del Modelo VAR ---")
print(model_fitted.summary())

# Este modelo se estima solo para ver como se comportarian las variables con 8 rezagos
model = VAR(df[['ln_PBI', 'ln_SP', 'ln_TCRM']])
lag_order = model.select_order(maxlags=8)
print(lag_order.summary())

"""
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
response = 'ln_SP'
h = irf.irfs.shape[0]

# Crear figura
fig, axes = plt.subplots(len(variables), 1, figsize=(8, 10), sharex=True)
fig.suptitle('Funciones Impulso-Respuesta para ln_SP', fontsize=18, weight='bold')

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
"""

################################################################################
# PRUEBA 1 - AUTOCORRELACION SERIAL DE LOS RESIDUOS (LGUN - BOX) 
################################################################################
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

# residuales en un DataFrame
resid = pd.DataFrame(model_fitted.resid, columns=model_fitted.names)

# test Ljung-Box para cada residuo (puedes cambiar lags)
lags = [4]

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
print("Raíces del polinomio AR:", np.round(roots, 4))

# Interpretación adicional
if np.all(np.abs(roots) < 1):
    print("✅ Todas las raíces están dentro del círculo unitario → el modelo es dinámicamente estable.")
else:
    print("⚠️ Algunas raíces están fuera del círculo unitario → el modelo es inestable.")
















"""
################################################################################
# PASO 6: FUNCIÓN IMPULSO-RESPUESTA (IRF)
################################################################################

# Generar las funciones impulso-respuesta (IRF)
irf = model_fitted.irf(8)  # horizonte de 8 períodos (puedes ajustar a 12 o más si deseas)
  
# Graficar solo las respuestas de ln_SP a los choques de todas las variables
irf.plot(orth=False, impulse=None, response='ln_SP')
plt.suptitle('Funciones Impulso-Respuesta para ln_SP', fontsize=14)
plt.show()
"""

"""
################################################################################
# PASO 5: ANÁLISIS POST-ESTIMACIÓN
################################################################################

# 5.1 Test de Causalidad de Granger
print("\n--- 7. Test de Causalidad de Granger ---")
causality_matrix = model_fitted.test_causality('ln_SP', df_log_diff.columns.drop('ln_SP'), kind='f')
print(causality_matrix.summary())

# 5.2 Funciones de Impulso-Respuesta (IRF)
print("\n--- 8. Generando Gráficos de Impulso-Respuesta ---")
irf = model_fitted.irf(10) # Analizar el efecto durante 10 periodos
irf.plot(orth=False, signif=0.05) # orth=False para shocks no ortogonalizados (Cholesky)
plt.suptitle('Funciones de Impulso-Respuesta (IRF) del Modelo VAR', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# 5.3 Descomposición de Varianza del Error de Pronóstico (FEVD)
print("\n--- 9. Generando Gráficos de Descomposición de Varianza (FEVD) ---")
fevd = model_fitted.fevd(10)
print(fevd.summary())
model_fitted.fevd(10).plot()
plt.suptitle('Descomposición de Varianza del Error de Pronóstico (FEVD)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("\nAnálisis VAR completado.")
"""


"""
################################################################################
# PASO 6: TEST DE CHOW PARA ESTABILIDAD ESTRUCTURAL
################################################################################
from scipy import stats

print("\n--- 10. Test de Chow para Estabilidad Estructural ---")

# Usaremos los datos diferenciados (estacionarios) para el test
data_chow = df_log_diff.copy()

# Definir el punto de quiebre (ej. inicio de la pandemia)
breakpoint_date = '2020-01-01'

# Definir variables para el modelo OLS del test
Y = data_chow['ln_SP']
X = data_chow.drop(columns=['ln_SP'])
X = sm.add_constant(X)

# Dividir en dos subperiodos
df1 = data_chow[data_chow.index < breakpoint_date]
df2 = data_chow[data_chow.index >= breakpoint_date]

Y1, X1 = df1['ln_SP'], sm.add_constant(df1.drop(columns=['ln_SP']))
Y2, X2 = df2['ln_SP'], sm.add_constant(df2.drop(columns=['ln_SP']))

# Ajustar los tres modelos: completo, subperiodo 1 y subperiodo 2
model_full = sm.OLS(Y, X).fit()
model1 = sm.OLS(Y1, X1).fit()
model2 = sm.OLS(Y2, X2).fit()

# Obtener los valores para la fórmula de Chow
k = X.shape[1]  # Número de parámetros (incluyendo la constante)
n1, n2 = len(df1), len(df2)
SSR_full = model_full.ssr
SSR1 = model1.ssr
SSR2 = model2.ssr

# Calcular el estadístico F de Chow
numerator = (SSR_full - (SSR1 + SSR2)) / k
denominator = (SSR1 + SSR2) / (n1 + n2 - 2 * k)
F_chow = numerator / denominator

# Calcular el p-valor
p_value = stats.f.sf(F_chow, k, n1 + n2 - 2 * k)

print(f"Punto de quiebre analizado: {breakpoint_date}")
print(f"F-statistic: {F_chow:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("\nResultado: Se rechaza la hipótesis nula (H0). Hay evidencia de un cambio estructural en el modelo.")
else:
    print("\nResultado: No se rechaza la hipótesis nula (H0). El modelo parece ser estructuralmente estable.")
"""