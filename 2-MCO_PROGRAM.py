import pandas as pd       # Manipulación y análisis de datos en tablas (DataFrames).
import numpy as np        # Cálculos numéricos y manejo de arreglos/matrices.
import seaborn as sns     # Visualización estadística avanzada (gráficos bonitos y rápidos).
import statsmodels.api as sm       # Modelos econométricos y estadísticos (regresiones, pruebas, etc.).
import statsmodels.stats.api as sms  # Pruebas estadísticas específicas (heterocedasticidad, autocorrelación...).
from statsmodels.stats.outliers_influence import variance_inflation_factor  
# Calcula el VIF (factor de inflación de la varianza) para detectar multicolinealidad.
import matplotlib.pyplot as plt    # Creación de gráficos básicos y personalizables.
from pathlib import Path           # Manejo de rutas y archivos de forma más segura y moderna.
from tabulate import tabulate      # Muestra tablas en consola con formato legible.

### Ver todas las hojas de excel ###
"""
excel = pd.ExcelFile('DATA/model_program.xlsx')
print(excel.sheet_names)  # Lista todas las hojas
"""
#### Ver que todas las hojas de excel y en cual se encuentra activa ###
"""
from openpyxl import load_workbook (modulo Workbook)
wb = load_workbook('DATA/model_program.xlsx')
print(wb.sheetnames)      # Muestra todas las hojas
print(wb.active.title)    # Muestra el nombre de la hoja activa
"""
################################################################################
# PASO 0: CONFIGURACIÓN Y CARGA DE DATOS
################################################################################

path = Path('DATA/model_program.xlsx')

# Cargar la hoja específica para la tesis
df = pd.read_excel(path, sheet_name='bd-tri', index_col=None)

# Limpiar nombres de columnas (buena práctica)
df.columns = df.columns.str.strip().str.replace(' ', '_')

#  Si la columna 'Año' está como índice, traerla de vuelta
if 'Año' not in df.columns and df.index.name == 'Año':
    df.reset_index(inplace=True)

# Renombrar columnas para trabajar más fácil
df = df.rename(columns={
    'Ingresos_Fiscales': 'Ingfisca',
})

print("--- 1. Datos Cargados y Preparados ---")
print(df.head())
print("\nInformación del DataFrame:")
df.info()

################################################################################
# PASO 1: Convertir la serie en logaritmos 
################################################################################

# Aplicar logaritmo natural (ln) a las variables positivas
df['ln_PBI'] = np.log(df['PBI'])
df['ln_Ingfisca'] = np.log(df['Ingfisca'])
df['ln_TIR'] = np.log(df['TIR'])
df['ln_TE'] = np.log(df['TE'])
df['ln_EP'] = np.log(df['EP'])

# Crear las diferencias logarítmicas (crecimientos porcentuales aproximados)
df['dln_PBI'] = df['ln_PBI'].diff()
df['dln_Ingfisca'] = df['ln_Ingfisca'].diff()
df['dln_TIR'] = df['ln_TIR'].diff()
df['dln_TE'] = df['ln_TE'].diff()
df['dln_EP'] = df['ln_EP'].diff()

# Crear variable de periodo trimestral
df['Fecha'] = pd.PeriodIndex(df['Año'], freq='Q')

# Crear dummy para quiebre estructural (ej. 2020Q3)
df['dummy_quiebre'] = ((df['Fecha'] >= '2020Q2')& (df['Fecha'] <= '2021Q2')).astype(int)

# (Opcional) Crear interacciones con las diferencias logarítmicas
df['dummy_dln_TIR'] = df['dummy_quiebre'] * df['dln_TIR']
df['dummy_dln_Ingfisca'] = df['dummy_quiebre'] * df['dln_Ingfisca']
df['dummy_dln_TE'] = df['dummy_quiebre'] * df['dln_TE']
df['dummy_dln_EP'] = df['dummy_quiebre'] * df['dln_EP']

# Eliminar los primeros NaN generados por la diferencia
df = df.dropna()

# Verificar
print(df[['Año', 'Fecha', 'dummy_quiebre']].tail(10))
print(df['dummy_quiebre'].value_counts())

################################################################################
# PASO 2: TEST DE ESTACIONARIEDAD (DICKEY-FULLER AUMENTADO)
################################################################################
from statsmodels.tsa.stattools import adfuller

def adf_test(series, name=''):
    "Realiza el test de Dickey-Fuller Aumentado en una serie temporal."
    result = adfuller(series.dropna())
    print(f'\n--- Test de Estacionariedad para: {name} ---')
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    if result[1] <= 0.05:
        print("✅ La serie es estacionaria.")
    else:
        print("❌ La serie no es estacionaria (tiene raíz unitaria).")

# Filtrar columnas con 'ln_'
cols_log = [col for col in df.columns if 'ln_' in col]

print("\n--- 2. Verificando Estacionariedad de las series logarítmicas ---")
for name in cols_log:
    adf_test(df[name], name=name)

###############################################################################
# PASO 3: PRUEBA DE CORRELACIÓN ENTRE VARIABLES
################################################################################

# Seleccionar variables que quieres correlacionar
cols = ['dln_PBI', 'dln_TIR', 'dln_Ingfisca', 'dln_TE']

# Calcular matriz de correlaciones
corr_matrix = df[cols].corr()

# Mostrar matriz en consola
print("\n=== Matriz de Correlaciones ===")
print(corr_matrix.round(3))

# Visualizar matriz con heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap de Correlaciones')
plt.show()
0.

###############################################################################
# PASO 4: MODELO MCO
################################################################################

# Definir variables explicativas y dependiente
Y = df['dln_PBI']
X = df[['dln_TIR', 'dln_Ingfisca', 'dln_TE', 'dln_EP', 
        'dummy_quiebre','dummy_dln_TIR','dummy_dln_Ingfisca']]
X = sm.add_constant(X)

# Ajustar modelo MCO simple
model = sm.OLS(Y, X).fit()
residuos = model.resid
print("\n=== RESULTADOS DEL MODELO MCO (Δln variables) ===")
print(model.summary())

# Crear tabla de coeficientes con intervalos de confianza
coef_table = pd.DataFrame({
    'Coeficiente': model.params,
    'Error Std': model.bse,
    't-stat': model.tvalues,
    'p-value': model.pvalues,
    'IC 0.025': model.conf_int()[0],
    'IC 0.975': model.conf_int()[1]
})

# Crear tabla resumen general del modelo
summary_table = pd.DataFrame({
    'Estadístico': ['R-squared', 'Adj. R-squared', 'F-statistic', 'Prob (F-statistic)',
                    'No. Observations', 'Log-Likelihood', 'AIC', 'BIC', 'Df Residuals', 'Df Model', 'Covariance Type'],
    'Valor': [model.rsquared, model.rsquared_adj, model.fvalue, model.f_pvalue,
              int(model.nobs), model.llf, model.aic, model.bic, model.df_resid, model.df_model, 'nonrobust']
})

# Mostrar todo con tabulate
print("\n=== Resumen General del Modelo ===")
print(tabulate(summary_table, headers='keys', tablefmt='fancy_grid', floatfmt=".4f"))

print("\n=== Coeficientes del Modelo ===")
print(tabulate(coef_table, headers='keys', tablefmt='fancy_grid', floatfmt=".6f"))

###############################################################################
# INTERPRETACIÓN AUTOMÁTICA CON ÍCONOS ✅❌
################################################################################
print("\n=== INTERPRETACIÓN DEL MODELO ===")

# R-cuadrado (bondad de ajuste)
if model.rsquared > 0.7:
    print(f"✅ El R² = {model.rsquared:.3f} indica que el modelo explica una proporción ALTA de la variabilidad del PBI.")
elif model.rsquared > 0.5:
    print(f"⚠️ El R² = {model.rsquared:.3f} indica una explicación MODERADA de la variabilidad del PBI.")
else:
    print(f"❌ El R² = {model.rsquared:.3f} sugiere un bajo poder explicativo; el modelo podría mejorarse.")

# Significancia global del modelo
if model.f_pvalue < 0.05:
    print(f"✅ La Prob(F) = {model.f_pvalue:.4f} < 0.05 indica que el modelo es GLOBALMENTE SIGNIFICATIVO.")
else:
    print(f"❌ La Prob(F) = {model.f_pvalue:.4f} > 0.05 indica que el modelo no es globalmente significativo.")

# Significancia individual de las variables
print("\n=== Variables estadísticamente significativas (p < 0.05) ===")
sig_vars = coef_table[coef_table['p-value'] < 0.05].index.tolist()
if sig_vars:
    print("✅ " + ", ".join(sig_vars))
else:
    print("❌ Ninguna variable es significativa al 5%.")

# Interpretación de la dummy de quiebre
if 'dummy_quiebre' in model.pvalues:
    if model.pvalues['dummy_quiebre'] < 0.05:
        print("\n✅ La variable 'dummy_quiebre' es significativa → evidencia un cambio estructural durante la pandemia.")
    else:
        print("\n❌ La variable 'dummy_quiebre' no es significativa → no se detecta un cambio estructural estadísticamente relevante.")

################################################################################
# PASO 3: PRUEBA DE MULTICOLINEALIDAD VIF
################################################################################

print("\n=== 5. Prueba de Multicolinealidad (VIF) ===")

# Usar las mismas variables explicativas del modelo MCO (sin la constante)
X_vif = X.drop(columns=['const'])

# Calcular VIF para cada variable
vif_data = pd.DataFrame({
    'Variable': X_vif.columns,
    'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})

# Mostrar tabla resumen
print("\n===Tabla resumen del VIF===")
print(tabulate(vif_data.round(3), headers='keys', tablefmt='fancy_grid', floatfmt=".4f"))

# Mostrar resultados con interpretación
for i in range(len(vif_data)):
    var = vif_data.loc[i, 'Variable']
    vif_val = vif_data.loc[i, 'VIF']

    if vif_val < 5:
        interpret = "✅ Bajo riesgo de multicolinealidad"
    elif vif_val < 10:
        interpret = "⚠️ Riesgo moderado de multicolinealidad"
    else:
        interpret = "❌ Alto riesgo de multicolinealidad"

    print(f"{var:>20}: VIF = {vif_val:.2f} → {interpret}")


################################################################################
# PASO 4: PRUEBA DE AUTOCORRELACION (DURBIN_WATSON) 
################################################################################

from statsmodels.stats.stattools import durbin_watson

print("\n=== 6. Prueba de Autocorrelación: Durbin-Watson ===")

dw = durbin_watson(model.resid)
dw_table = pd.DataFrame({
    'Estadístico': ['Durbin-Watson'],
    'Valor': [dw]
})

print(tabulate(dw_table, headers='keys', tablefmt='fancy_grid', floatfmt=".4f"))

# Interpretación académica
if 1.5 <= dw <= 2.5:
    interpretacion = "✅ No se evidencia autocorrelación en los residuos (resultado deseable)."
elif dw < 1.5:
    interpretacion = "⚠️ Indicio de autocorrelación positiva en los residuos."
else:
    interpretacion = "⚠️ Indicio de autocorrelación negativa en los residuos."

print(f"Interpretación: {interpretacion}")

# Interpretación:
# DW ≈ 2.0: No hay autocorrelación (ideal).
# DW < 2.0: Posible autocorrelación positiva.
# DW > 2.0: Posible autocorrelación negativa.

################################################################################
# PASO 5: PRUEBA DE HETEROCEDASTICIDAD (TEST DE WHITE)
################################################################################

from statsmodels.stats.diagnostic import het_white
# Test de White
white_test = het_white(model.resid, model.model.exog)

white_test_table = pd.DataFrame({
    'Estadístico': ['LM stat', 'LM p-value', 'F-stat', 'F p-value'],
    'Valor': white_test
})

print("\n=== Resultado del Test de White (Heterocedasticidad) ===")
print(tabulate(white_test_table, headers='keys', tablefmt='fancy_grid', floatfmt=".4f"))

if white_test[1] > 0.05:
    print("✅ No se rechaza H0: la varianza de los errores es constante (homocedástica)")
else:
    print("⚠️ Se rechaza H0: la varianza de los errores no es constante (heterocedástica)")

# H0 (Hipótesis nula): La varianza de los errores es constante (homocedasticidad)
# H1 (Hipótesis alternativa): La varianza de los errores no es constante (heterocedasticidad)

# Nota: Si el p-value del test es mayor que el nivel de significancia (por ejemplo, 0.05),
# no se rechaza H0, lo que indica que no hay evidencia significativa de heterocedasticidad"""


################################################################################
# PASO 6: PRUEBA DE NORMALIDAD EN LOS RESIDUOS (JARQUE - BERA)
################################################################################

from statsmodels.stats.stattools import jarque_bera

jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuos)

jb_table = pd.DataFrame({
    'Estadístico': ['JB estadístico', 'p-value', 'Skew', 'Kurtosis'],
    'Valor': [jb_stat, jb_pvalue, skew, kurtosis]
})

print("\n=== Jarque - Bera ===")
print(tabulate(jb_table, headers='keys', tablefmt='fancy_grid', floatfmt=".4f"))

if jb_pvalue > 0.05:
    print("✅ No se rechaza H0: los residuos se distribuyen normalmente")
else:
    print("⚠️ Se rechaza H0: los residuos no son normales")


################################################################################
# PASO 8: ESTABILIDAD ESTRUCTURAL (TEST DE CHOW)
################################################################################
from scipy import stats

def chow_test(df, split_index):
    """Realiza el test de Chow en el punto de quiebre indicado (por posición)"""
    
    # Dividir usando posición, no etiquetas (iloc)
    Y1 = df.iloc[:split_index]['dln_PBI']
    X1 = sm.add_constant(df.iloc[:split_index][['dln_TIR', 'dln_Ingfisca', 'dln_TE', 'dln_EP',
        'dummy_quiebre', 'dummy_dln_TIR', 'dummy_dln_Ingfisca', 'dummy_dln_TE', 'dummy_dln_EP']])
    
    Y2 = df.iloc[split_index:]['dln_PBI']
    X2 = sm.add_constant(df.iloc[split_index:][['dln_TIR', 'dln_Ingfisca', 'dln_TE', 'dln_EP',
        'dummy_quiebre', 'dummy_dln_TIR', 'dummy_dln_Ingfisca', 'dummy_dln_TE', 'dummy_dln_EP']])
    
    # Modelo completo
    Y_full = df['dln_PBI']
    X_full = sm.add_constant(df[[ 'dln_TIR', 'dln_Ingfisca', 'dln_TE', 'dln_EP',
        'dummy_quiebre', 'dummy_dln_TIR', 'dummy_dln_Ingfisca', 'dummy_dln_TE', 'dummy_dln_EP']])
    
    model_full = sm.OLS(Y_full, X_full).fit()
    model1 = sm.OLS(Y1, X1).fit()
    model2 = sm.OLS(Y2, X2).fit()
    
    # Estadístico F de Chow
    n1, n2 = len(Y1), len(Y2)
    k = X_full.shape[1]
    SSR_full = sum(model_full.resid ** 2)
    SSR1 = sum(model1.resid ** 2)
    SSR2 = sum(model2.resid ** 2)
    
    F = ((SSR_full - (SSR1 + SSR2)) / k) / ((SSR1 + SSR2) / (n1 + n2 - 2 * k))
    p_value = 1 - stats.f.cdf(F, k, n1 + n2 - 2 * k)
    return F, p_value

# EVALUAR TODOS LOS POSIBLES PUNTOS DE QUIEBRE
# Reiniciamos el índice para que el loop funcione bien (Año pasa a columna normal)

df_reset = df.reset_index(drop=False).rename(columns={'index': 'Trimestre'})

results = []
for i in range(8, len(df_reset) - 8):  # evita cortes con pocas observaciones
    F, p = chow_test(df_reset, i)
    results.append((df_reset.loc[i, 'Año'], F, p))

results_df = pd.DataFrame(results, columns=['Trimestre', 'F_stat', 'p_value'])

# MOSTRAR RESULTADOS

best_break = results_df.loc[results_df['F_stat'].idxmax()]

print("📊 Test de Chow — Resultados por Trimestre")
print(results_df.to_string(index=False))
print("\n🏆 Posible punto de cambio estructural:")
print(best_break)

if best_break['p_value'] < 0.05:
    print(f"\n❌ Se rechaza H₀: cambio estructural detectado en el trimestre {best_break['Trimestre']}")
else:
    print(f"\n✅ No se rechaza H₀: el modelo es estable estructuralmente")


