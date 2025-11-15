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
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from scipy.stats import norm
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import norm
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


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

path = Path('DATA/BD_SUBNA.xlsx')

# Cargar la hoja específica para la tesis
df = pd.read_excel(path, sheet_name='base_subna', index_col=None)

# Limpiar nombres de columnas (buena práctica)
df.columns = df.columns.str.strip().str.replace(' ', '_')

#  Si la columna 'Año' está como índice, traerla de vuelta
if 'Año' not in df.columns and df.index.name == 'Año':
    df.reset_index(inplace=True)

# Renombrar columnas para trabajar más fácil
df = df.rename(columns={
    'Ingresos_propios_perc': 'Ing_pro_per',
})

print("--- 1. Datos Cargados y Preparados ---")
print(df.head())
print("\nInformación del DataFrame:")
df.info()

################################################################################
# PASO 1: Creación de fecha y Dummy de quiebre 
################################################################################

# Crear variable de periodo trimestral
df['Fecha'] = pd.PeriodIndex(df['Año'], freq='Q')

# Crear dummy para quiebre estructural (ej. 2020Q3)
df['dummy_quiebre'] = ((df['Fecha'] >= pd.Period('2020Q2')) & 
                       (df['Fecha'] <= pd.Period('2021Q2'))).astype(int)

# Verificar
print(df[['Año', 'Fecha', 'dummy_quiebre']].tail(10))
print(df['dummy_quiebre'].value_counts())

################################################################################
# PASO 2: TEST DE ESTACIONARIEDAD (ADF)
################################################################################

def adf_test(series, name=''):
    """Ejecuta el test Dickey-Fuller Aumentado con limpieza automática."""
    
    # Asegurar que la serie sea numérica
    s = pd.to_numeric(series, errors='coerce').dropna()
    
    if len(s) < 5:
        print(f"\n--- Test ADF: {name} ---")
        print("⚠️ Serie insuficiente para ADF (menos de 5 datos).")
        return

    result = adfuller(s)

    print(f'\n--- Test ADF: {name} ---')
    print(f'Estadístico ADF: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('Valores críticos:')
    for key, value in result[4].items():
        print(f'   {key}: {value:.4f}')

    if result[1] <= 0.05:
        print("✅ Serie estacionaria (rechaza raíz unitaria).")
    else:
        print("❌ Serie no estacionaria (no rechaza raíz unitaria).")

# TOMAR SOLO LAS COLUMNAS NUMÉRICAS DEL DATAFRAME

df_numeric = df.select_dtypes(include=['int', 'float'])

print("\n--- 3. Estacionariedad en series originales (todas las numéricas) ---")
for col in df_numeric.columns:
    adf_test(df_numeric[col], col)

# PRIMERAS DIFERENCIAS

df_numeric_diff = df_numeric.diff().dropna()

print("\n--- 4. Estacionariedad en primeras diferencias (numéricas) ---")
for col in df_numeric_diff.columns:
    adf_test(df_numeric_diff[col], col + "_diff")

################################################################################
# PASO 3: PRUEBA DE CORRELACIÓN ENTRE VARIABLES (NIVELES)
################################################################################

import seaborn as sns
import matplotlib.pyplot as plt

# Seleccionar variables que quieres correlacionar
cols = ['saldo_perc_pob', 'Ing_pro_per', 'gasto_cap_perc','transf_corr_pc']

# Verificar que todas las columnas existan en el DataFrame
missing = [c for c in cols if c not in df.columns]
if missing:
    print("⚠️ Las siguientes columnas no existen en el DataFrame:", missing)
else:
    # Filtrar solo valores numéricos
    df_corr = df[cols].apply(pd.to_numeric, errors='coerce')

    # Calcular matriz de correlaciones
    corr_matrix = df_corr.corr()

    # Mostrar matriz en consola
    print("\n=== Matriz de Correlaciones (Niveles) ===")
    print(corr_matrix.round(3))

    # Visualizar matriz con heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Heatmap de Correlaciones - Niveles')
    plt.tight_layout()
    plt.show()

# Crear DataFrame solo con las columnas seleccionadas
df_vars = df[cols].apply(pd.to_numeric, errors='coerce')

# Crear primeras diferencias
df_diff_vars = df_vars.diff().dropna()

# Renombrar columnas: saldo_perc_pob -> d_saldo_perc_pob
df_diff_vars = df_diff_vars.rename(columns=lambda x: f"d_{x}")

################################################################################
# PASO 3B: GENERAR DIFERENCIAS Y CORRELACIONARLAS
################################################################################

# Matriz de correlaciones para diferencias
corr_matrix_diff = df_diff_vars.corr()

print("\n=== Matriz de Correlaciones (Primeras Diferencias) ===")
print(corr_matrix_diff.round(3))

# Heatmap para diferencias
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix_diff, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap de Correlaciones - Primeras Diferencias')
plt.tight_layout()
plt.show()

###############################################################################
# PASO 4: MODELO MCO
################################################################################

# Agregar dummy a las diferencias (nota: dummy no se diferencia)
df_diff_vars['dummy_quiebre'] = df['dummy_quiebre'].iloc[1:].values

# Definir variables explicativas y dependiente
Y = df_diff_vars['d_saldo_perc_pob']
X = df_diff_vars[['d_Ing_pro_per', 'd_gasto_cap_perc','dummy_quiebre','d_transf_corr_pc']]
X = sm.add_constant(X)

# Ajustar modelo MCO simple
model = sm.OLS(Y, X).fit()
residuos = model.resid
print("\n=== RESULTADOS DEL MODELO MCO (Δln variables) ===")
print(model.summary())

# Ajustar modelo MCO con errores robustos HAC (Newey-West)
model_hac = sm.OLS(Y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 2})

print("\n=== RESULTADOS DEL MODELO MCO CON HAC (NEWEY–WEST) ===")
print(model_hac.summary())
# maxlags = número de rezagos permitidos en la estructura de autocorrelación
model_hac = sm.OLS(Y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 2})


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
# INTERPRETACIÓN AUTOMÁTICA 
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

jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(model.resid)

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

print("\n=== 8. Test de Chow: Estabilidad Estructural ===")

# 1. Determinar el punto de quiebre (2020Q2)
break_period = pd.Period('2020Q2', freq='Q')

# df_diff_vars empieza en t=2, por eso usamos df['Fecha'].iloc[1:]
break_index = df_diff_vars.index[df['Fecha'].iloc[1:] == break_period][0]

print(f"\n📌 Punto de quiebre evaluado: {break_period} (índice {break_index})")

# 2. Dividir la muestra
Y1 = Y.loc[:break_index]
X1 = X.loc[:break_index]

Y2 = Y.loc[break_index+1:]
X2 = X.loc[break_index+1:]

# 3. Estimar submodelos
model1 = sm.OLS(Y1, X1).fit()
model2 = sm.OLS(Y2, X2).fit()

# 4. SSR
SSR1 = sum(model1.resid**2)
SSR2 = sum(model2.resid**2)
SSR_full = sum(model.resid**2)

# 5. Parámetros
k = X.shape[1]  # número de parámetros estimados (incluye constante)
n1, n2 = len(Y1), len(Y2)

# 6. Estimador F de Chow
F_chow = ((SSR_full - (SSR1 + SSR2)) / k) / ((SSR1 + SSR2) / (n1 + n2 - 2*k))
p_value_chow = 1 - stats.f.cdf(F_chow, k, (n1 + n2 - 2*k))

# 7. Tabla de resultados
chow_table = pd.DataFrame({
    'Estadístico': ['F-statistic', 'p-value', 'SSR_full', 'SSR1', 'SSR2', 'k', 'n1', 'n2'],
    'Valor': [F_chow, p_value_chow, SSR_full, SSR1, SSR2, k, n1, n2]
})

print("\n=== Resultados del Test de Chow ===")
print(tabulate(chow_table, headers='keys', tablefmt='fancy_grid', floatfmt=".6f"))

# 8. Conclusión
if p_value_chow < 0.05:
    print(f"\n❌ Se rechaza H₀: existe evidencia de cambio estructural en {break_period}.")
else:
    print(f"\n✅ No se rechaza H₀: el modelo es estable, no se detecta cambio estructural en {break_period}.")


################################################################################
# PASO 9: ESPECIFICACIÓN FUNCIONAL (TEST RAMSEY RESET)
################################################################################
from statsmodels.stats.diagnostic import linear_reset

print("\n=== 9. Test de Especificación Funcional: Ramsey RESET ===")

reset_test = linear_reset(model, power=2, use_f=True)
print(f"Estadístico F: {reset_test.fvalue:.4f}")
print(f"p-value: {reset_test.pvalue:.4f}")

if reset_test.pvalue < 0.05:
    print("❌ Se rechaza H₀: posible forma funcional incorrecta o variables omitidas.")
    print("➡️ Considera incluir términos no lineales, interacciones o transformaciones.")
else:
    print("✅ No se rechaza H₀: la forma funcional del modelo parece adecuada.")

################################################################################
# PASO 10: ROBUSTEZ DE INFERENCIA (ERRORES HAC)
################################################################################
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.stats.sandwich_covariance import se_cov

print("\n=== 12. Robustez de Inferencia: Errores HAC (Newey-West) ===")

# Calcular errores estándar robustos tipo HAC
cov_hac_matrix = cov_hac(model, nlags=2)
se_hac = se_cov(cov_hac_matrix)

hac_table = pd.DataFrame({
    'Coeficiente': model.params,
    'Error Std (HAC)': se_hac,
    't-HAC': model.params / se_hac
})

print(tabulate(hac_table.round(6), headers='keys', tablefmt='fancy_grid'))
print("✅ Se aplicaron errores estándar robustos (Newey-West) para verificar estabilidad de inferencia.")

print("\n💡 Recomendación: Si los signos y significancia de los coeficientes se mantienen similares,")
print("puedes concluir que los resultados del modelo son robustos frente a heterocedasticidad o autocorrelación leve.")

################################################################################
# PASO 11: GRÁFICOS COMPLEMENTARIOS DEL MODELO MCO
################################################################################

################################################################################
# PASO 11: GRÁFICOS COMPLEMENTARIOS DEL MODELO MCO
################################################################################

# Preparar eje temporal para gráficos (usar timestamp para matplotlib)
# si Fecha es PeriodIndex, convertir a timestamp para graficar
if isinstance(df['Fecha'].iloc[0], pd.Period):
    df['Fecha_plot'] = df['Fecha'].dt.to_timestamp()
    # df_diff_vars está alineado con df.iloc[1:]
    df_diff_vars['Fecha_plot'] = df['Fecha_plot'].iloc[1:].values
else:
    df['Fecha_plot'] = pd.to_datetime(df['Fecha'])
    df_diff_vars['Fecha_plot'] = df['Fecha_plot'].iloc[1:].values

# GRAFICO GENERAL DE SERIES (DIFERENCIAS)
plt.figure(figsize=(12,6))
plt.plot(df_diff_vars['Fecha_plot'], df_diff_vars['d_Ing_pro_per'], label="Δ Ing_pro_per", linewidth=2)
plt.plot(df_diff_vars['Fecha_plot'], df_diff_vars['d_gasto_cap_perc'], label="Δ gasto_cap_perc", linewidth=2)
plt.plot(df_diff_vars['Fecha_plot'], df_diff_vars['d_transf_corr_pc'], label="Δ transf_corr_pc", linewidth=2)
# Dummy como puntos escalados para visualización
scale = df_diff_vars[['d_Ing_pro_per', 'd_gasto_cap_perc', 'd_transf_corr_pc']].abs().max().max()
plt.scatter(df_diff_vars['Fecha_plot'], df_diff_vars['dummy_quiebre'] * scale,
            label="Dummy quiebre (escalada)", color="red", zorder=5)
plt.title("Evolución de las series en diferencias")
plt.xlabel("Fecha")
plt.ylabel("Delta (primeras diferencias)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Gráficos adicionales útiles
# Residuos vs Ajustados
plt.figure(figsize=(8,5))
plt.scatter(model.fittedvalues, residuos)
plt.axhline(0, linestyle='--', color='k')
plt.xlabel("Valores ajustados")
plt.ylabel("Residuos")
plt.title("Residuos vs Valores ajustados")
plt.tight_layout()
plt.show()

# Histograma residuos + curva normal
plt.figure(figsize=(8,5))
plt.hist(residuos, bins=15, density=True, alpha=0.6)
xs = np.linspace(residuos.min(), residuos.max(), 200)
plt.plot(xs, norm.pdf(xs, residuos.mean(), residuos.std()))
plt.title("Histograma de residuos con curva normal")
plt.tight_layout()
plt.show()

# QQ-plot
qqplot(residuos, line='45')
plt.title("QQ-Plot de residuos")
plt.tight_layout()
plt.show()

# ACF residuos
plot_acf(residuos)
plt.title("ACF de residuos")
plt.tight_layout()
plt.show()

# Coeficientes estandarizados (betas)
betas_std = model.params.copy()
# estandarizar: beta * sd(X)/sd(Y) para variables no constantes
sd_X = X.drop(columns=['const']).std()
for name in sd_X.index:
    betas_std[name] = model.params[name] * (sd_X[name] / Y.std())
# dejar const como NaN para mostrar solo explicativas
betas_plot = betas_std.drop('const', errors='ignore')
plt.figure(figsize=(8,5))
betas_plot.plot(kind='bar')
plt.title("Coeficientes estandarizados (aprox.)")
plt.tight_layout()
plt.show()

################################################################################
# PASO 12: TABLA DE ELASTICIDADES E INTERPRETACIÓN
################################################################################

elasticidades = pd.DataFrame({
    'Variable': model.params.index,
    'Coeficiente (β)': model.params.values,
})

# Agregar medias (X sin const)
media_Y = Y.mean()
medias_X = X.drop(columns=['const']).mean()
# vectorizar medias (poner NaN para const)
elasticidades['Media X'] = [medias_X.get(v, np.nan) if v != 'const' else np.nan for v in elasticidades['Variable']]
elasticidades['Media Y'] = media_Y

elasticidades['Elasticidad'] = elasticidades.apply(
    lambda r: (r['Coeficiente (β)'] * r['Media X'] / r['Media Y']) if pd.notna(r['Media X']) else np.nan,
    axis=1
)

print("\n=== TABLA DE ELASTICIDADES ===")
print(tabulate(elasticidades, headers='keys', tablefmt='fancy_grid', floatfmt=".6f"))

print("\n=== INTERPRETACIÓN DE LAS ELASTICIDADES ===")
for _, row in elasticidades.iterrows():
    var = row['Variable']
    elast = row['Elasticidad']
    if pd.isna(elast):
        continue
    print(f"- Un cambio relativo del 1% en {var.replace('d_','')} se asocia con un cambio aproximado de {elast:.3f}% en la variable dependiente.")

################################################################################
# FIN DEL SCRIPT CORREGIDO
################################################################################

##########################################################
# GRAFICO: VALOR REAL VS VALOR PREDICHO
##########################################################

plt.figure(figsize=(10,6))

# Valores reales
plt.plot(df["Fecha"], df["d_Ing_pro_per"], label="Real", linewidth=2)

# Valores predichos del MCO
plt.plot(df["Fecha"], results.fittedvalues, label="Predicho (MCO)", linewidth=2, linestyle="--")

plt.title("Comparación entre valores reales y predichos", fontsize=14)
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
















