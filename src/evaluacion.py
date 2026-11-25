"""Métricas comunes a todos los modelos. La clase positiva es phishing (1)."""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def metricas(y_real, prob_phishing, umbral: float = 0.5) -> dict:
    """Calcula las métricas a partir de la probabilidad de phishing.

    `prob_phishing` es la salida del modelo entre 0 y 1; si supera el umbral
    (0,5 por defecto) la URL se clasifica como phishing.
    """
    prob_phishing = np.asarray(prob_phishing).ravel()
    pred = (prob_phishing >= umbral).astype(int)
    vn, fp, fn, vp = confusion_matrix(y_real, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_real, pred),
        "precision": precision_score(y_real, pred, zero_division=0),
        "recall": recall_score(y_real, pred),
        "f1": f1_score(y_real, pred),
        "roc_auc": roc_auc_score(y_real, prob_phishing),
        "VP": int(vp), "FN": int(fn), "FP": int(fp), "VN": int(vn),
    }


def tabla(filas: list) -> pd.DataFrame:
    """Convierte una lista de dicts de métricas en una tabla ordenada."""
    return pd.DataFrame(filas)
