"""
Demo: clasifica una URL como phishing o legítima.

Uso:
    python demo.py                                   # modo interactivo
    python demo.py "http://secure-login-verify.xyz"  # una URL
    python demo.py --comparar "https://www.google.com/maps"

Por defecto usa el mejor modelo que solo mira el DOMINIO (sin atajos), elegido
en notebooks/04_evaluacion_final.ipynb. Con --comparar muestra también el
modelo entrenado con la URL original, que aprendió los atajos del dataset.

La demo NO visita la URL: solo analiza el texto. Es segura con cualquier enlace.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
from src.texto import CodificadorCaracteres, dominio_de_url  # noqa: E402

DIR_MOD = RAIZ / "models"

# Nombre del modelo (como aparece en el notebook 04) -> archivos
MODELOS_DOMINIO = {
    "CNN · dominio": ("cnn_dominio.keras", "codificador_dominio.json"),
    "BiGRU · dominio": ("bigru_dominio.keras", "codificador_dominio.json"),
    "TF-IDF + RL · dominio": ("tfidf_dominio.joblib", None),
}


def cargar_modelo(archivo, codificador):
    """Devuelve una función texto -> probabilidad de phishing."""
    if archivo.endswith(".joblib"):
        import joblib
        modelo = joblib.load(DIR_MOD / archivo)
        return lambda t: float(modelo.predict_proba([t])[0, 1])

    from tensorflow import keras
    modelo = keras.models.load_model(DIR_MOD / archivo)
    cod = CodificadorCaracteres.cargar(DIR_MOD / codificador)
    return lambda t: float(modelo.predict(cod.transformar([t]), verbose=0)[0, 0])


def barra(prob, ancho=30):
    llenos = round(prob * ancho)
    return "[" + "#" * llenos + "." * (ancho - llenos) + "]"


def mostrar(nombre, prob, umbral):
    veredicto = "PHISHING" if prob >= umbral else "legítima"
    print(f"  {nombre:28s} {barra(prob)} {prob * 100:5.1f} %  -> {veredicto}")


def main():
    parser = argparse.ArgumentParser(description="Detector de URLs de phishing")
    parser.add_argument("url", nargs="?", help="URL a analizar (si se omite, modo interactivo)")
    parser.add_argument("--modelo", choices=list(MODELOS_DOMINIO),
                        help="modelo de dominio a usar (por defecto, el mejor según el notebook 04)")
    parser.add_argument("--umbral", type=float, default=0.5,
                        help="probabilidad desde la cual se considera phishing (por defecto 0,5)")
    parser.add_argument("--comparar", action="store_true",
                        help="mostrar también el modelo con atajos (URL original)")
    args = parser.parse_args()

    config = json.loads((RAIZ / "reports" / "resultados" / "modelo_demo.json").read_text())
    nombre = args.modelo or config["modelo"]
    print(f"Cargando modelo: {nombre} ...")
    predecir_dominio = cargar_modelo(*MODELOS_DOMINIO[nombre])
    predecir_url = None
    if args.comparar:
        predecir_url = cargar_modelo("cnn_url_original.keras", "codificador_url_original.json")

    def analizar(url):
        dominio = dominio_de_url(url)
        if not dominio:
            print("  No se pudo extraer un dominio de esa URL.")
            return
        print(f"\nURL:     {url}\nDominio: {dominio}")
        mostrar(nombre, predecir_dominio(dominio), args.umbral)
        if predecir_url:
            mostrar("CNN · URL original (atajos)", predecir_url(url.strip().lower()), args.umbral)

    if args.url:
        analizar(args.url)
        return

    print("Escriba una URL y presione Enter (vacío para salir).")
    while True:
        try:
            url = input("\nURL> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not url:
            break
        analizar(url)


if __name__ == "__main__":
    main()
