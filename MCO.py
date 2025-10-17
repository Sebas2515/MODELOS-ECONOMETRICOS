
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from pathlib import Path
from tabulate import tabulate


################################################################################
#                           MCO
################################################################################

# 1️⃣ Cargar datos
path = Path('DATA/bd-subnacionales.xlsx')
df = pd.read_excel(path)

# 2️⃣ Limpiar nombres de columnas (quitar espacios al inicio/final y saltos de línea)
df.columns = df.columns.str.strip().str.replace('\n','').str.replace('\r','')

print(df)

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































"""
# 3️⃣ Renombrar columnas para trabajar más fácil
df = df.rename(columns={
    'TASA % EJECUCIÓN PRESUPUESTAL (avance)': 'Ejecucion',
    'Capacidad recaudatoria total': 'Capacidad_recaudatoria',
    'Recursos directamente recaudados': 'RDR'
})

# Aplicar logaritmo natural (ln) a las variables positivas
df['ln_PIM'] = np.log(df['PIM'])
df['ln_Capacidad_recaudatoria'] = np.log(df['Capacidad_recaudatoria'])


# 3️⃣ Definir variables
Y = df['Ejecucion']  # variable dependiente
X = df[['ln_PIM','ln_Capacidad_recaudatoria','RDR']]  # variables explicativas
X = sm.add_constant(X)  # agregar intercepto

# 4️⃣ Ajustar modelo MCO simple

model = sm.OLS(Y, X).fit()
residuos = model.resid
"""
"""
# 6️⃣ Crear tabla de coeficientes con intervalos de confianza
coef_table = pd.DataFrame({
    'Coeficiente': model.params,
    'Error Std': model.bse,
    't-stat': model.tvalues,
    'p-value': model.pvalues,
    'IC 0.025': model.conf_int()[0],
    'IC 0.975': model.conf_int()[1]
})

# 7️⃣ Crear tabla resumen general del modelo
summary_table = pd.DataFrame({
    'Estadístico': ['R-squared', 'Adj. R-squared', 'F-statistic', 'Prob (F-statistic)',
                    'No. Observations', 'Log-Likelihood', 'AIC', 'BIC', 'Df Residuals', 'Df Model', 'Covariance Type'],
    'Valor': [model.rsquared, model.rsquared_adj, model.fvalue, model.f_pvalue,
              int(model.nobs), model.llf, model.aic, model.bic, model.df_resid, model.df_model, 'nonrobust']
})

# 8️⃣ Mostrar todo con tabulate
print("\n=== Resumen General del Modelo ===")
print(tabulate(summary_table, headers='keys', tablefmt='fancy_grid', floatfmt=".4f"))

print("\n=== Coeficientes del Modelo ===")
print(tabulate(coef_table, headers='keys', tablefmt='fancy_grid', floatfmt=".6f"))

"""
########################## PRUEBA DE CORRELACIÓN ##########################

"""
# 5️⃣ Seleccionar variables que quieres correlacionar
cols = ['Ejecucion', 'RDR', 'ln_PIM', 'ln_Capacidad_recaudatoria']

# 6️⃣ Calcular matriz de correlaciones
corr_matrix = df[cols].corr()

# 7️⃣ Mostrar matriz en consola
print("\n=== Matriz de Correlaciones ===")
print(corr_matrix.round(3))

# 8️⃣ Visualizar matriz con heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap de Correlaciones')
plt.show()
"""

########################## PRUEBA DE AUTOCORRELACION (DURBIN_WATSON) ##########################
"""
from statsmodels.stats.stattools import durbin_watson

dw = durbin_watson(model.resid)
print('Durbin-Watson:', dw)
"""
# Interpretación:
# DW ≈ 2.0: No hay autocorrelación (ideal).
# DW < 2.0: Posible autocorrelación positiva.
# DW > 2.0: Posible autocorrelación negativa.

########################## PRUEBA DE HETEROCEDASTICIDAD (TEST DE WHITE) ##########################
"""
from statsmodels.stats.diagnostic import het_white
# Test de White
white_test = het_white(model.resid, model.model.exog)

# Etiquetas
labels = ['LM stat', 'LM p-value', 'F-stat', 'F p-value']

# Crear un DataFrame para mostrarlo bonito
white_df = pd.DataFrame([white_test], columns=labels)

print("📊 Resultado del Test de White (Heterocedasticidad)")
print(white_df)

# H0 (Hipótesis nula): La varianza de los errores es constante (homocedasticidad)
# H1 (Hipótesis alternativa): La varianza de los errores no es constante (heterocedasticidad)

# Nota: Si el p-value del test es mayor que el nivel de significancia (por ejemplo, 0.05),
# no se rechaza H0, lo que indica que no hay evidencia significativa de heterocedasticidad.
"""

########################## PRUEBA DE MULTICOLINEALIDAD VIF ##########################
"""
#Calcular VIF
vif_data = pd.DataFrame()
vif_data['Variable'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)
"""


########################## PRUEBA DE NORMALIDAD EN LOS RESIDUOS (JARQUE - BERA) ##########################
"""
from statsmodels.stats.stattools import jarque_bera

jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuos)

print("Jarque-Bera Test")
print("JB estadístico:", jb_stat)
print("p-value:", jb_pvalue)
print("Skew:", skew)
print("Kurtosis:", kurtosis)

if jb_pvalue > 0.05:
    print("No se rechaza H0: los residuos se distribuyen normalmente")
else:
    print("Se rechaza H0: los residuos no son normales")
"""

########################## PRUEBA DE ESTABILIDAD ESTRUCTURAL (JARQUE - BERA) ##########################
"""
from scipy import stats

# Supongamos que 'df' tiene tu data de 2014-2024
# Dividir en dos subperiodos
df1 = df[df['AÑO'] <= 2018]
df2 = df[df['AÑO'] > 2018]

Y1 = df1['Ejecucion'] 
X1 = sm.add_constant(df1[['ln_PIM', 'ln_Capacidad_recaudatoria', 'RDR']])

Y2 = df2['Ejecucion'] 
X2 = sm.add_constant(df2[['ln_PIM', 'ln_Capacidad_recaudatoria', 'RDR']])

# Ajustar los modelos
model_full = sm.OLS(Y, X).fit()
model1 = sm.OLS(Y1, X1).fit()
model2 = sm.OLS(Y2, X2).fit()

# Número de observaciones y parámetros
n1 = len(Y1)
n2 = len(Y2)
k = X.shape[1]  # número de parámetros incluyendo constante

# Sum of Squared Residuals
SSR_full = sum(model_full.resid ** 2)
SSR1 = sum(model1.resid ** 2)
SSR2 = sum(model2.resid ** 2)

# Estadístico F de Chow
numerador = (SSR_full - (SSR1 + SSR2)) / k
denominador = (SSR1 + SSR2) / (n1 + n2 - 2*k)
F_chow = numerador / denominador

# p-value
p_value = 1 - stats.f.cdf(F_chow, k, n1 + n2 - 2*k)

print("📊 Test de Chow (estabilidad estructural)")
print("F estadístico:", F_chow)
print("p-value:", p_value)

if p_value < 0.05:
    print("Se rechaza H0: hay evidencia de cambio estructural")
else:
    print("No se rechaza H0: el modelo es estructuralmente estable")
"""

"""
# --- 1. Autocorrelación de los residuos (Durbin-Watson) ---
print("\n--- PRUEBA DE AUTOCORRELACIÓN (Durbin-Watson) ---")
dw_stats = durbin_watson(model_fitted.resid)
for col, val in zip(model_fitted.names, dw_stats):
    print(f"{col}: {val:.2f} {'→ posible autocorrelación' if (val < 1.5 or val > 2.5) else '→ sin autocorrelación aparente'}")
"""



















































"""
# 5️⃣ Resultados
print(model.summary())
"""
"""

# 6️⃣ Guardar predicciones
df_model['Y_pred'] = model.predict(X)

# 7️⃣ Gráficos de Ejecucion real vs predicha
outdir = Path('DATA/analisis_outputs')
outdir.mkdir(parents=True, exist_ok=True)

plt.figure()
plt.plot(df_model['AÑO'], df_model['Ejecucion'], marker='o', label='Real')
plt.plot(df_model['AÑO'], df_model['Y_pred'], marker='x', label='Predicha')
plt.title('Ejecución presupuestal: Real vs Predicha')
plt.xlabel('Año')
plt.ylabel('Ejecución (0-1)')
plt.legend()
plt.grid(True)
plt.savefig(outdir/'ejecucion_vs_predicha.png')
plt.close()

# 8️⃣ Guardar datos con predicciones
df_model.to_csv(outdir/'df_model_simple.csv', index=False)

print('Análisis completo. Resultados y gráficos guardados en:', outdir)
"""






"""
# Convertir Ejecución de % a decimal
df['Ejecucion'] = df['Ejecucion'].str.replace('%','').astype(float)/100
"""
"""
# Eliminar filas con datos faltantes en variables que usaremos
df_model = df.dropna(subset=['Ejecucion','PIM','Capacidad_recaudatoria','RDR']).copy()
"""



