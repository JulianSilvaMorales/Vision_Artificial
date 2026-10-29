"""Graficas comunes: curvas de entrenamiento y matrices de confusion."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from src.utils.config import CLASES

ETIQUETAS = ["Degeneracion", "Normal", "Protrusion"]


def matriz_confusion(y, yp, titulo, ruta, normalizar=False):
    cm = confusion_matrix(y, yp, labels=range(len(CLASES)))
    ver = cm / cm.sum(1, keepdims=True) if normalizar else cm
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(ver, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(CLASES)), ETIQUETAS, fontsize=8)
    ax.set_yticks(range(len(CLASES)), ETIQUETAS, fontsize=8)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]}\n({ver[i, j]:.0%})" if normalizar else str(cm[i, j])
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if ver[i, j] > ver.max() / 2 else "black")
    ax.set_title(titulo, fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def curvas(hist, ruta, titulo):
    """hist: DataFrame con epoca, train_loss, val_loss, train_acc, val_acc, val_f1_macro."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    axs[0].plot(hist.epoca, hist.train_loss, "o-", label="train")
    axs[0].plot(hist.epoca, hist.val_loss, "o-", label="val")
    axs[0].set_title("Perdida (entropia cruzada)")
    axs[0].set_xlabel("Epoca")
    axs[0].legend()
    axs[1].plot(hist.epoca, hist.train_acc, "o-", label="exactitud train")
    axs[1].plot(hist.epoca, hist.val_acc, "o-", label="exactitud val")
    axs[1].plot(hist.epoca, hist.val_f1_macro, "o-", label="F1 macro val")
    axs[1].set_title("Exactitud y F1")
    axs[1].set_xlabel("Epoca")
    axs[1].legend()
    fig.suptitle(titulo, fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
