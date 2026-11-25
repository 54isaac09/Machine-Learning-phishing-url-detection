"""
Variables léxicas propias, calculadas SOLO a partir del dominio.

Las variables precalculadas de PhiUSIIL se obtuvieron de la URL completa, así
que "heredan" los atajos del dataset (https://www., ausencia de ruta). Para
tener un escenario tabular sin atajos (A4), calculamos nuestras propias
variables mirando únicamente el dominio, por ejemplo 'secure-login.xyz'.
"""

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

# Palabras frecuentes en dominios de phishing (ingeniería social)
PALABRAS_SOSPECHOSAS = [
    "login", "signin", "secure", "account", "verify", "update", "confirm",
    "bank", "pay", "wallet", "support", "service", "mail", "web", "auth",
]

# TLD (terminación del dominio) más comunes. Se codifican con 0/1 (one-hot);
# cualquier otro TLD cae en "tld_otro". La lista es fija, no aprendida de los
# datos, para no filtrar información del conjunto de test.
TLDS_COMUNES = ["com", "org", "net", "edu", "gov", "de", "uk", "io", "app",
                "dev", "xyz", "top", "ru", "info", "br", "cn"]

_VOCALES = set("aeiou")
_RE_IP = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def _entropia(texto: str) -> float:
    """Entropía de Shannon: qué tan 'aleatorio' es el texto.

    Dominios generados al azar (ej. 'x7k2q9f.top') tienen entropía alta.
    """
    if not texto:
        return 0.0
    n = len(texto)
    return -sum(c / n * math.log2(c / n) for c in Counter(texto).values())


def _racha_maxima(texto: str, conjunto) -> int:
    """Largo de la secuencia más larga de caracteres que pertenecen a `conjunto`."""
    mejor = actual = 0
    for ch in texto:
        actual = actual + 1 if ch in conjunto else 0
        mejor = max(mejor, actual)
    return mejor


_DIGITOS = set("0123456789")
_CONSONANTES = set("bcdfghjklmnpqrstvwxyz")


def variables_lexicas(dominios: pd.Series) -> pd.DataFrame:
    """Devuelve una tabla con ~40 variables numéricas por dominio."""
    d = dominios.fillna("").str.lower()
    partes = d.str.split(".")
    tld = partes.str[-1]
    nombre = partes.apply(lambda p: max(p[:-1], key=len) if len(p) > 1 else p[0])

    X = pd.DataFrame(index=d.index)
    X["largo"] = d.str.len()
    X["n_puntos"] = d.str.count(r"\.")
    X["n_guiones"] = d.str.count("-")
    X["n_digitos"] = d.str.count(r"\d")
    X["prop_digitos"] = X.n_digitos / X.largo.clip(lower=1)
    letras = d.str.count(r"[a-z]")
    X["prop_vocales"] = d.apply(lambda s: sum(ch in _VOCALES for ch in s)) / letras.clip(lower=1)
    X["entropia"] = d.apply(_entropia)
    X["racha_digitos"] = d.apply(lambda s: _racha_maxima(s, _DIGITOS))
    X["racha_consonantes"] = d.apply(lambda s: _racha_maxima(s, _CONSONANTES))
    X["largo_tld"] = tld.str.len()
    X["largo_parte_principal"] = nombre.str.len()
    X["es_ip"] = d.str.match(_RE_IP).astype(int)
    X["tiene_punycode"] = d.str.contains("xn--", regex=False).astype(int)

    for palabra in PALABRAS_SOSPECHOSAS:
        X[f"palabra_{palabra}"] = d.str.contains(palabra, regex=False).astype(int)
    X["n_palabras_sospechosas"] = X.filter(like="palabra_").sum(axis=1)

    for t in TLDS_COMUNES:
        X[f"tld_{t}"] = (tld == t).astype(int)
    X["tld_otro"] = (~tld.isin(TLDS_COMUNES)).astype(int)
    return X.astype(float)
