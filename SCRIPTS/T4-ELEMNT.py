# ===========================================================
# SIMULACIÓN DEL MODELO DE VOLATILIDAD ESTOCÁSTICA DE HESTON
# Empresa: LIMA CAPITAL INVESTMENTS S.A.C.
# Autor: [Tu nombre]
# Objetivo: Valorar una call europea por Monte Carlo
# ===========================================================

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. PARÁMETROS DEL MODELO
# -----------------------------
S0 = 50        # Precio inicial del activo
v0 = 0.04      # Varianza inicial (0.04 -> 20% de volatilidad)
r = 0.04       # Tasa libre de riesgo
mu = 0.06      # Tasa esperada del activo
kappa = 2.0    # Velocidad de reversión de la varianza
theta = 0.04   # Varianza de largo plazo
sigma_v = 0.5  # Volatilidad de la varianza
rho = -0.6     # Correlación entre shocks
T = 1.0        # Horizonte temporal (1 año)
K = 52         # Strike de la opción

# -----------------------------
# 2. PARÁMETROS DE SIMULACIÓN
# -----------------------------
N = 250        # Número de pasos (días hábiles)
M = 10000      # Número de trayectorias Monte Carlo
dt = T / N     # Paso temporal

# -----------------------------
# 3. INICIALIZACIÓN DE MATRICES
# -----------------------------
S = np.zeros((N + 1, M))
v = np.zeros((N + 1, M))
S[0, :] = S0
v[0, :] = v0

# -----------------------------
# 4. SIMULACIÓN DEL PROCESO
# -----------------------------
np.random.seed(42)  # Fijar semilla para reproducibilidad

for t in range(1, N + 1):
    Z1 = np.random.randn(M)
    Z2 = np.random.randn(M)
    Z2_cor = rho * Z1 + np.sqrt(1 - rho**2) * Z2  # correlación

    # Actualizar varianza (Euler con truncamiento)
    v_prev = np.maximum(v[t - 1, :], 0)
    v[t, :] = v_prev + kappa * (theta - v_prev) * dt + sigma_v * np.sqrt(v_prev * dt) * Z2_cor
    v[t, :] = np.maximum(v[t, :], 0)

    # Actualizar precio (log-Euler)
    S[t, :] = S[t - 1, :] * np.exp((mu - 0.5 * v_prev) * dt + np.sqrt(v_prev * dt) * Z1)

# -----------------------------
# 5. PRECIO DE LA CALL EUROPEA
# -----------------------------
ST = S[-1, :]
payoffs = np.maximum(ST - K, 0)
C0 = np.exp(-r * T) * np.mean(payoffs)
stderr = np.exp(-r * T) * np.std(payoffs) / np.sqrt(M)

# -----------------------------
# 6. RESULTADOS
# -----------------------------
print("==========================================")
print("SIMULACIÓN MODELO DE HESTON")
print("Empresa: LIMA CAPITAL INVESTMENTS S.A.C.")
print("==========================================")
print(f"Precio estimado de la call: {C0:.4f}")
print(f"Error estándar: {stderr:.4f}")
print(f"Varianza media final: {np.mean(v[-1, :]):.4f}")
print("==========================================")

# -----------------------------
# 7. GRÁFICOS
# -----------------------------
# Graficar algunas trayectorias
plt.figure(figsize=(8, 4))
for i in range(10):
    plt.plot(np.linspace(0, T, N + 1), S[:, i])
plt.title("Simulación de precios - Modelo de Heston")
plt.xlabel("Tiempo (años)")
plt.ylabel("Precio del activo S_t")
plt.grid(True)
plt.show()

# Histograma de precios finales
plt.figure(figsize=(7, 4))
plt.hist(ST, bins=50, color='skyblue', edgecolor='black')
plt.title("Distribución de precios finales S_T")
plt.xlabel("S_T")
plt.ylabel("Frecuencia")
plt.show()
