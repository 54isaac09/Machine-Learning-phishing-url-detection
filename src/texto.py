"""
Herramientas para los modelos que leen el texto de la URL carácter por carácter.

- `dominio_de_url`: extrae el dominio de cualquier URL que escriba el usuario
  (lo usa la demo).
- `CodificadorCaracteres`: convierte texto en secuencias de números enteros de
  largo fijo, que es lo que reciben las redes neuronales (CNN y BiGRU).
"""

import json
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np


def dominio_de_url(url: str) -> str:
    """'HTTPS://www.Ejemplo.com:8080/login?x=1' -> 'ejemplo.com'."""
    url = url.strip()
    if "://" not in url:
        url = "http://" + url  # sin esquema, urlsplit no reconoce el dominio
    host = urlsplit(url).hostname or ""
    host = host.lower()
    return host[4:] if host.startswith("www.") else host


class CodificadorCaracteres:
    """Asigna un número a cada carácter y rellena con ceros hasta `largo_max`.

    0 = relleno (padding), 1 = carácter desconocido, 2.. = caracteres vistos en
    entrenamiento. Ejemplo: si se ajusta con ['ab.cl'] y largo_max=6, el
    vocabulario es {'.': 2, 'a': 3, 'b': 4, 'c': 5, 'l': 6} y
    'ab.cl' -> [3, 4, 2, 5, 6, 0]; un carácter nunca visto ('z') -> 1.
    """

    def __init__(self, largo_max: int = 64):
        self.largo_max = largo_max
        self.indice = {}

    def ajustar(self, textos):
        """Construye el vocabulario SOLO con los textos de entrenamiento."""
        caracteres = sorted(set("".join(textos)))
        self.indice = {ch: i + 2 for i, ch in enumerate(caracteres)}
        return self

    @property
    def tam_vocabulario(self) -> int:
        return len(self.indice) + 2

    def transformar(self, textos) -> np.ndarray:
        X = np.zeros((len(textos), self.largo_max), dtype=np.int32)
        for fila, texto in enumerate(textos):
            codigos = [self.indice.get(ch, 1) for ch in texto[: self.largo_max]]
            X[fila, : len(codigos)] = codigos
        return X

    def guardar(self, ruta):
        Path(ruta).write_text(json.dumps({"largo_max": self.largo_max,
                                          "indice": self.indice}, ensure_ascii=False))

    @classmethod
    def cargar(cls, ruta):
        datos = json.loads(Path(ruta).read_text())
        cod = cls(datos["largo_max"])
        cod.indice = datos["indice"]
        return cod
