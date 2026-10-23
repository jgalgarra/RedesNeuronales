"""
Red neuronal 2 → 3 → 1 implementada con PyTorch sin clases de optimización.
Mismo problema que bp_hardwired_ReLU.ipynb: clasificar puntos dentro/fuera
de un círculo de radio 0.7.

La actualización de pesos se hace a mano:  p ← p − (lr / tam_batch) · ∇p
sin usar ningún torch.optim.*.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import random

# ─── Datos ────────────────────────────────────────────────────────────────────

def crea_datos(num_puntos):
    """Genera puntos (x, y) en [0,1]² con etiqueta 1 si x²+y² < 0.7²."""
    x = np.random.rand(num_puntos).astype(np.float32)
    y = np.random.rand(num_puntos).astype(np.float32)
    etiquetas = ((x**2 + y**2) < 0.7**2).astype(np.float32)
    return x, y, etiquetas

def mostrar_datos(x, y, labels):
    """Dibuja los puntos coloreados por clase (azul=0, verde=1, rojo=error)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    for clase, color in enumerate(["blue", "green", "red"]):
        mask = labels == clase
        ax.scatter(x[mask], y[mask], color=color, s=10, zorder=50)
    ax.set_aspect("equal", "datalim")
    plt.show()

# ─── Conjunto de datos ────────────────────────────────────────────────────────

NUM_PUNTOS_ALL   = 10000
NUM_PUNTOS_TRAIN = round(0.8 * NUM_PUNTOS_ALL)
NUM_PUNTOS_TEST  = NUM_PUNTOS_ALL - NUM_PUNTOS_TRAIN

x_tr, y_tr, e_tr = crea_datos(NUM_PUNTOS_TRAIN)
x_te, y_te, e_te = crea_datos(NUM_PUNTOS_TEST)

# ─── Parámetros de la red ─────────────────────────────────────────────────────
# Convenio: w[l] ∈ ℝ^(n_l × n_{l-1})  →  z[l] = w[l] · a[l-1] + b[l]

n0, n1, n2 = 2, 3, 1

W1 = torch.randn(n1, n0) * (2 / n0) ** 0.5   # He    — capa oculta (ReLU),     shape (3,2)
b1 = torch.zeros(n1)                            # shape (3,)
W2 = torch.randn(n2, n1) * (1 / n1) ** 0.5   # Xavier — capa de salida (sigmoid), shape (1,3)
b2 = torch.zeros(n2)                            # shape (1,)

parametros = [W1, b1, W2, b2]
for p in parametros:
    p.requires_grad_(True)

# ─── Propagación hacia adelante ───────────────────────────────────────────────

def forward(entrada):
    """Calcula la salida de la red para un vector de entrada de tamaño 2."""
    a1 = torch.relu(W1 @ entrada + b1)      # capa oculta:   ReLU(W1·x + b1)
    a2 = torch.sigmoid(W2 @ a1 + b2)        # capa de salida: σ(W2·a1 + b2)
    return a2

# ─── Evaluación ───────────────────────────────────────────────────────────────

def evaluar_clasificacion(x, y, etiquetas, num_puntos):
    """Clasifica todas las muestras con umbral 0.5 y devuelve aciertos y predicciones."""
    aciertos     = 0
    predicciones = np.zeros(num_puntos)
    with torch.no_grad():
        for i in range(num_puntos):
            entrada = torch.tensor([x[i], y[i]])
            pred    = float(forward(entrada)[0] > 0.5)
            predicciones[i] = pred
            aciertos += (pred == etiquetas[i])
    return aciertos, predicciones

aciertos_ini, _ = evaluar_clasificacion(x_tr, y_tr, e_tr, NUM_PUNTOS_TRAIN)
print(f"Tasa inicial (train): {aciertos_ini / NUM_PUNTOS_TRAIN:.4f}")

# ─── Entrenamiento: SGD manual por mini-batches ───────────────────────────────

tam_batch        = 10
num_epocas       = 200
tasa_aprendizaje = 0.2
indices          = list(range(NUM_PUNTOS_TRAIN))
historial_coste  = []

for epoca in range(num_epocas):
    # Mezclar los datos al inicio de cada época
    random.shuffle(indices)
    x_tr = x_tr[indices]
    y_tr = y_tr[indices]
    e_tr = e_tr[indices]

    coste_epoca = 0.0
    num_batches = NUM_PUNTOS_TRAIN // tam_batch

    # Acumulador de gradientes para toda la época
    acum = {p: torch.zeros_like(p.data) for p in parametros}

    for idx_batch in range(num_batches):
        grad_batch = {p: torch.zeros_like(p.data) for p in parametros}

        for idx_muestra in range(tam_batch):
            k        = idx_batch * tam_batch + idx_muestra
            entrada  = torch.tensor([x_tr[k], y_tr[k]])
            etiqueta = torch.tensor([e_tr[k]])

            # Paso 1: propagación hacia adelante
            pred = forward(entrada)

            # Paso 2: coste C = ½(pred − y)²
            coste = 0.5 * (pred - etiqueta) ** 2
            coste_epoca += coste.item()

            # Paso 3: retropropagación (PyTorch calcula los gradientes)
            coste.backward()

            # Acumular gradiente de esta muestra en el batch
            for p in parametros:
                grad_batch[p] = grad_batch[p] + p.grad.detach().clone()
                p.grad.zero_()

        # Sumar el gradiente del batch al acumulador de la época
        for p in parametros:
            acum[p] = acum[p] + grad_batch[p]

        # Actualizar pesos a mano, sin ningún optimizador
        for p in parametros:
            p.data -= (tasa_aprendizaje / tam_batch) * acum[p]

    coste_medio = coste_epoca / NUM_PUNTOS_TRAIN
    historial_coste.append(coste_medio)
    if epoca % 10 == 0:
        print(f"Época {epoca:3d} — Pérdida media: {coste_medio:.4f}")

# ─── Curva de aprendizaje ─────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(historial_coste, color="steelblue", linewidth=1.5)
ax.set_xlabel("Época")
ax.set_ylabel("Pérdida media")
ax.set_title("Curva de aprendizaje (PyTorch manual)")
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# ─── Resultados en test ───────────────────────────────────────────────────────

aciertos_test, predicciones_test = evaluar_clasificacion(x_te, y_te, e_te, NUM_PUNTOS_TEST)
print(f"Tasa de acierto en test: {aciertos_test / NUM_PUNTOS_TEST:.4f}")

# Marcar errores con clase 2 (rojo) para visualización
nplot     = min(1000, NUM_PUNTOS_TEST)
pred_plot = predicciones_test[:nplot].copy()
for i in range(nplot):
    if pred_plot[i] != e_te[i]:
        pred_plot[i] = 2

mostrar_datos(x_te[:nplot], y_te[:nplot], pred_plot)
