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

# 1. Creamos el DataFrame de residuos
residuals = pd.DataFrame(
    vecm_fitted.resid,
    columns=df_vecm.columns,
    # Mantenemos el índice numérico por defecto, pero alineado al final del df_vecm
    index=df_vecm.index[-vecm_fitted.resid.shape[0]:] 
)

# ==============================================================================
# 🌟 LÓGICA COMPACTA: Generación y Sincronización de Etiquetas
# ==============================================================================
n_resid_periods = residuals.shape[0]
n_total_periods = len(df_vecm)
n_lags_omitted = n_total_periods - n_resid_periods

# Generamos el PeriodIndex completo (asumiendo 2016Q1)
full_period_index = pd.period_range(start='2016Q1', periods=n_total_periods, freq='Q').astype(str)

# Tomamos el segmento de índice que corresponde a los residuos
resid_labels = full_period_index[n_lags_omitted:].tolist() 
resid_positions = np.arange(n_resid_periods)
# ==============================================================================


# 2. Gráfico de residuos por variable
ax_list = residuals.plot(subplots=True, figsize=(12, 10), title="Residuos del modelo VECM")

# Ajustar el eje X para que muestre el rango completo
for ax in ax_list:
    # 🌟 FORZAMOS POSICIONES Y ETIQUETAS
    ax.set_xticks(resid_positions)
    ax.set_xticklabels(resid_labels, rotation=45, ha='right')
    
    # 🌟 CORRECCIÓN CLAVE: Mostrar solo cada N-ésimo tick para evitar recorte y superposición
    # Mantenemos solo una etiqueta cada 4 trimestres (inicio de año) para mayor legibilidad
    for i, label in enumerate(ax.get_xticklabels()):
        # Oculta las etiquetas que NO son el inicio de un bloque anual
        if i % 4 != 0: 
            label.set_visible(False)
    
    # Aseguramos que el eje X se extienda a todo el rango de datos
    ax.set_xlim(resid_positions[0] - 0.5, resid_positions[-1] + 0.5) 

    ax.axhline(0, color='grey', linestyle='--') 

plt.tight_layout()
plt.show()

################################################################################
# PRUEBA 1 - AUTOCORRELACION SERIAL DE LOS RESIDUOS (LGUN - BOX) 
################################################################################
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

# residuales en un DataFrame
resid = pd.DataFrame(vecm_fitted.resid, columns=vecm_fitted.names)

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
"""
rint("\n--- PRUEBA DE ESTABILIDAD DEL MODELO VAR ---")

stable = vecm_fitted.is_stable()
print(f"¿El modelo es estable?: {'✅ Sí' if stable else '❌ No'}")

roots = vecm_fitted.roots
print("Raíces del polinomio AR:", np.round(roots, 4))

# Interpretación adicional
if np.all(np.abs(roots) < 1):
    print("✅ Todas las raíces están dentro del círculo unitario → el modelo es dinámicamente estable.")
else:
    print("⚠️ Algunas raíces están fuera del círculo unitario → el modelo es inestable.")
"""

################################################################################
# PRUEBA DE ESTABILIDAD DEL MODELO VECM (Solución Definitiva con .beta_matrices)
################################################################################
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------
# ⚠️ ADJUNTE ESTE BLOQUE INMEDIATAMENTE DESPUÉS DE: vecm_fitted = vecm_model.fit()
# --------------------------------------------------------------------------------

print("\n--- PRUEBA DE ESTABILIDAD DEL MODELO VECM (Método Final) ---")

try:
    # Parámetros del modelo
    k_endog = vecm_fitted.neqs    # k = 6
    k_ar = vecm_fitted.k_ar       # p = 4

    # 1. Extraer las matrices A_i del VAR subyacente
    # Este atributo contiene [A1, A2, ..., Ap]. Es un array de arrays (4 x 6 x 6)
    A_matrices = vecm_fitted.beta_matrices 

    # 2. Construir la Matriz Companion C ((k*p) x (k*p)) -> (24 x 24)
    Companion_matrix = np.zeros((k_endog * k_ar, k_endog * k_ar))
    
    # Fila superior: [A1, A2, A3, A4]
    Companion_matrix[:k_endog, :] = np.hstack(A_matrices)
    
    # Bloque inferior: Matriz Identidad desplazada I_{k(p-1)}
    Companion_matrix[k_endog:, :-k_endog] = np.eye(k_endog * (k_ar - 1))

    # 3. Calcular valores propios y evaluar estabilidad
    eigvals = np.linalg.eigvals(Companion_matrix)
    abs_eigvals = np.abs(eigvals)
    max_root = np.max(abs_eigvals)

    print(f"\nDimensión Matriz Companion: {Companion_matrix.shape}")
    print(f"Valor propio con mayor módulo (máx. raíz): {max_root:.4f}")

    if max_root < 1.0001:
        print("✅ El modelo VECM es estable.")
        print("   (La raíz máxima es cercana a 1.0, lo cual es esperado por la cointegración).")
    else:
        print(f"⚠️ El modelo VECM NO es estable (máximo módulo: {max_root:.4f} > 1.0).")

    # --- Gráfico de Raíces ---
    plt.figure(figsize=(6, 6))
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', label='Círculo unitario')
    plt.gca().add_artist(circle)
    plt.scatter(eigvals.real, eigvals.imag, color='blue', label='Raíces del VECM')
    plt.title("Estabilidad del VECM (Matriz Companion Final)")
    plt.xlabel("Parte Real")
    plt.ylabel("Parte Imaginaria")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

except Exception as e:
    print(f"❌ Error Irresoluble: La extracción con '.beta_matrices' falló. {e}")
    print("El error de tipado persiste. Sugerencia: Intente ajustar el VECM sin el término determinístico: deterministic=None, para aislar el error.")