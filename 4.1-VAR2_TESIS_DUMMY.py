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

# ✅ Crear dummy para quiebre estructural (ej. 2021Q3)
df['dummy_quiebre'] = (df.index >= '2021Q3').astype(int)

# Verificar
print(df[['dummy_quiebre']].tail(10))
print(df['dummy_quiebre'].value_counts())

# Crear un nuevo DataFrame limpio (sin valores nulos)

df_clean = df.dropna().copy()

print("\n--- 2. DataFrame Limpio (sin valores nulos) ---")
print(df_clean.head())
print("\nNúmero de observaciones finales:", len(df_clean))

# Confirmar nombres finales de columnas

print("\nColumnas finales disponibles:")
print(df_clean.columns.tolist())

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
# PASO 2: SELECCIÓN DEL ORDEN DE REZAGOS (LAGS) ÓPTIMO
################################################################################
from statsmodels.tsa.api import VAR

model = VAR(df_n_diff)
print("\n--- 4. Selección de Rezagos Óptimos (AIC, BIC, FPE, HQIC) ---")
lag_selection = model.select_order(maxlags=4)
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
endog_vars = df[['N_S&P', 'N_TIR','N_TCRM','N_IPC','N_PBI']]
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


















################################################################################
# PASO 9: PRUEBA DE QUIEBRE ESTRUCTURAL (CHOW) 
################################################################################
"""
# PASO 9.1: CREAR VARIABLES DIFERENCIADAS

df_diff = df_clean.diff().dropna()
df_diff.columns = [col + "_diff" for col in df_diff.columns]

print("\n--- 2️⃣ Columnas diferenciadas disponibles ---")
print(df_diff.columns.tolist())

# PASO 9.2: DEFINIR FUNCIÓN DEL TEST DE CHOW

def chow_test(df, split_index, dep_var, indep_vars):
    Realiza el test de Chow para detectar quiebres estructurales
    
    Y1 = df.iloc[:split_index][dep_var]
    X1 = sm.add_constant(df.iloc[:split_index][indep_vars])
    
    Y2 = df.iloc[split_index:][dep_var]
    X2 = sm.add_constant(df.iloc[split_index:][indep_vars])
    
    Y_full = df[dep_var]
    X_full = sm.add_constant(df[indep_vars])
    
    model_full = sm.OLS(Y_full, X_full).fit()
    model1 = sm.OLS(Y1, X1).fit()
    model2 = sm.OLS(Y2, X2).fit()
    
    n1, n2 = len(Y1), len(Y2)
    k = X_full.shape[1]
    SSR_full = sum(model_full.resid ** 2)
    SSR1 = sum(model1.resid ** 2)
    SSR2 = sum(model2.resid ** 2)
    
    F = ((SSR_full - (SSR1 + SSR2)) / k) / ((SSR1 + SSR2) / (n1 + n2 - 2 * k))
    p_value = 1 - stats.f.cdf(F, k, n1 + n2 - 2 * k)
    return F, p_value


# PASO 9.3: EVALUAR POSIBLES QUIEBRES

# Variable dependiente: PBI en diferencia logarítmica
dep_var = "N_PBI_diff"

# Variables explicativas (ajusta según tu modelo base)
indep_vars = ["N_TIR_diff", "N_TCRM_diff", "N_IPC_diff", "N_S&P_diff"]

# Filtramos solo las columnas necesarias
df_quiebre = df_diff[[dep_var] + indep_vars].dropna()

results = []
for i in range(8, len(df_quiebre) - 8):  # evita cortes extremos
    F, p = chow_test(df_quiebre, i, dep_var, indep_vars)
    results.append((df_quiebre.index[i], F, p))

results_df = pd.DataFrame(results, columns=['Periodo', 'F_stat', 'p_value'])


# PASO 9.4: MOSTRAR RESULTADOS


best_break = results_df.loc[results_df['F_stat'].idxmax()]
print("\n📊 Test de Chow — Resultados por Periodo")
print(results_df.to_string(index=False))
print("\n🏆 Posible punto de quiebre estructural:")
print(best_break)

if best_break['p_value'] < 0.05:
    print(f"\n❌ Se rechaza H₀ → Cambio estructural detectado en {best_break['Periodo']}")
else:
    print(f"\n✅ No se rechaza H₀ → El modelo es estable estructuralmente")

"""
"""
################################################################################
# PASO FINAL: EXPORTAR RESULTADOS A WORD (.docx)
################################################################################
from docx import Document
from docx.shared import Inches
import matplotlib.pyplot as plt
import io

# Crear documento
doc = Document()
doc.add_heading('📊 Resultados del Modelo VAR con Dummy Exógena', level=1)

# --- 1. Información general ---
doc.add_heading('1️⃣ Información del Modelo', level=2)
doc.add_paragraph(f"Lags óptimos (según AIC): {optimal_lags}")
doc.add_paragraph(f"Variables incluidas: {', '.join(model_fitted.names)}")
doc.add_paragraph("Variable exógena incluida: Dummy de quiebre estructural (2021Q3 en adelante)")

# --- 2. Resumen del modelo VAR ---
doc.add_heading('2️⃣ Resumen del Modelo VAR', level=2)
summary_text = str(model_fitted.summary())
doc.add_paragraph(summary_text)

# --- 3. Correlaciones ---
doc.add_heading('3️⃣ Matriz de Correlación entre Variables', level=2)
corr_buf = io.StringIO()
corr_matrix.to_string(corr_buf)
doc.add_paragraph(corr_buf.getvalue())

# --- 4. Multicolinealidad (VIF) ---
doc.add_heading('4️⃣ Prueba de Multicolinealidad (VIF)', level=2)
for i in range(len(vif_data)):
    var = vif_data.loc[i, 'Variable']
    vif_val = vif_data.loc[i, 'VIF']
    if vif_val < 5:
        interpret = "✅ Bajo riesgo de multicolinealidad"
    elif vif_val < 10:
        interpret = "⚠️ Riesgo moderado de multicolinealidad"
    else:
        interpret = "❌ Alto riesgo de multicolinealidad"
    doc.add_paragraph(f"{var}: VIF = {vif_val:.2f} → {interpret}", style='List Bullet')

# --- 5. Causalidad de Granger ---
doc.add_heading('5️⃣ Prueba de Causalidad de Granger', level=2)
doc.add_paragraph("Hipótesis nula (H₀): la variable de la fila NO causa a la variable de la columna.")
granger_buf = io.StringIO()
results_df.to_string(granger_buf)
doc.add_paragraph(granger_buf.getvalue())

# --- 6. Estabilidad del Modelo VAR ---
doc.add_heading('6️⃣ Estabilidad del Modelo VAR', level=2)
roots = model_fitted.roots
stable = np.all(np.abs(roots) > 1)
doc.add_paragraph(f"Raíces del polinomio AR: {', '.join([f'{r:.3f}' for r in roots])}")
if stable:
    doc.add_paragraph("✅ El modelo es estable (todas las raíces están fuera del círculo unitario).")
else:
    doc.add_paragraph("⚠️ El modelo NO es estable (algunas raíces dentro del círculo unitario).")

# --- 7. Diagnóstico de los residuos ---
doc.add_heading('7️⃣ Pruebas de Diagnóstico de Residuos', level=2)

# Autocorrelación (Ljung–Box)
doc.add_paragraph("🔹 Prueba de Autocorrelación (Ljung–Box):")
for col in resid.columns:
    lb = acorr_ljungbox(resid[col], lags=[optimal_lags], return_df=True)
    p_value = lb['lb_pvalue'].iloc[-1]
    result = f"{col}: p-value = {p_value:.4f} → "
    result += "❌ Autocorrelación presente" if p_value < 0.05 else "✅ Sin autocorrelación"
    doc.add_paragraph(result, style='List Bullet')

# Heterocedasticidad (ARCH)
doc.add_paragraph("🔹 Prueba de Heterocedasticidad (ARCH):")
for col in resid.columns:
    f_stat, f_pvalue, lm_stat, lm_pvalue = het_arch(resid[col])
    result = f"{col}: F p-value = {f_pvalue:.4f}, LM p-value = {lm_pvalue:.4f} → "
    result += "✅ Varianza constante" if f_pvalue > 0.05 and lm_pvalue > 0.05 else "⚠️ Heterocedasticidad detectada"
    doc.add_paragraph(result, style='List Bullet')

# Normalidad (Jarque–Bera)
doc.add_paragraph("🔹 Prueba de Normalidad (Jarque–Bera):")
for col in resid.columns:
    jb = stats.jarque_bera(resid[col])
    jb_stat, jb_pvalue = jb.statistic, jb.pvalue
    result = f"{col}: JB = {jb_stat:.3f}, p-value = {jb_pvalue:.4f} → "
    result += "✅ Normalidad no rechazada" if jb_pvalue > 0.05 else "⚠️ Residuos no normales"
    doc.add_paragraph(result, style='List Bullet')

# --- 8. Funciones Impulso-Respuesta (solo N_S&P como variable de respuesta) ---
doc.add_heading('8️⃣ Funciones Impulso-Respuesta (N_S&P)', level=2)
irf = model_fitted.irf(8)
response = 'N_S&P'
variables = model_fitted.names
h = irf.irfs.shape[0]

for var in variables:
    if var != response:
        irf_line = irf.irfs[:, variables.index(var), variables.index(response)]
        se = irf.stderr()[..., variables.index(var), variables.index(response)]
        lower, upper = irf_line - 2 * se, irf_line + 2 * se

        # Graficar y guardar cada IRF
        plt.figure(figsize=(6, 4))
        plt.plot(irf_line, label=f'Choque en {var}', lw=2.2)
        plt.fill_between(np.arange(h), lower, upper, alpha=0.2)
        plt.axhline(0, color='black', lw=1, linestyle='--')
        plt.title(f"Respuesta de N_S&P ante un choque en {var}")
        plt.xlabel("Horizonte (periodos)")
        plt.ylabel("Respuesta")
        plt.legend()
        plt.tight_layout()

        img_stream = io.BytesIO()
        plt.savefig(img_stream, format='png', bbox_inches='tight')
        plt.close()
        img_stream.seek(0)
        doc.add_picture(img_stream, width=Inches(6))

# --- 9. Prueba de Quiebre Estructural (Chow) ---
doc.add_heading('9️⃣ Prueba de Quiebre Estructural (Chow)', level=2)
best_break_text = f"Periodo con mayor F: {best_break['Periodo']}, F = {best_break['F_stat']:.3f}, p-value = {best_break['p_value']:.4f}"
doc.add_paragraph(best_break_text)
if best_break['p_value'] < 0.05:
    doc.add_paragraph(f"❌ Se rechaza H₀ → Cambio estructural detectado en {best_break['Periodo']}")
else:
    doc.add_paragraph("✅ No se rechaza H₀ → No se detecta quiebre estructural significativo.")

# --- 10. Guardar documento ---
output_path = "Resultados_VAR_Completo.docx"
doc.save(output_path)
print(f"✅ Archivo Word generado correctamente: {output_path}")
"""

#pruebas de ensayo y error 

#ya me aburriiii xd
#etiquetar siempre las funciones 
