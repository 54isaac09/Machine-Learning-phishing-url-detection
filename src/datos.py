"""
Funciones de carga, limpieza y división del dataset PhiUSIIL.

Todos los notebooks usan este módulo para que los datos y la división
train/val/test sean siempre los mismos (reproducibilidad).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]
DIR_DATOS = RAIZ / "data"
RUTA_CSV = DIR_DATOS / "PhiUSIIL_Phishing_URL_Dataset.csv"
RUTA_DIVISION = DIR_DATOS / "division.csv"

SEMILLA = 42

# --------------------------------------------------------------------------
# Grupos de columnas
# --------------------------------------------------------------------------
# Variables calculadas solo a partir del texto de la URL
COLS_URL = [
    "URLLength", "DomainLength", "IsDomainIP", "TLDLength", "NoOfSubDomain",
    "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio",
    "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL",
    "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS",
    "CharContinuationRate", "TLDLegitimateProb", "URLCharProb",
]

# Variable derivada que compara la URL con una lista de URLs legítimas.
# En este dataset separa las clases casi perfectamente (ver 01_eda.ipynb).
COLS_SOSPECHOSAS = ["URLSimilarityIndex"]

# Variables que requieren descargar y analizar el HTML de la página
COLS_PAGINA = [
    "LineOfCode", "LargestLineLength", "HasTitle", "DomainTitleMatchScore",
    "URLTitleMatchScore", "HasFavicon", "Robots", "IsResponsive",
    "NoOfURLRedirect", "NoOfSelfRedirect", "HasDescription", "NoOfPopup",
    "NoOfiFrame", "HasExternalFormSubmit", "HasSocialNet", "HasSubmitButton",
    "HasHiddenFields", "HasPasswordField", "Bank", "Pay", "Crypto",
    "HasCopyrightInfo", "NoOfImage", "NoOfCSS", "NoOfJS", "NoOfSelfRef",
    "NoOfEmptyRef", "NoOfExternalRef",
]

# Columnas de texto (no se usan directamente como variables numéricas)
COLS_TEXTO = ["URL", "Domain", "TLD", "Title"]

OBJETIVO = "es_phishing"  # 1 = phishing, 0 = legítima


# --------------------------------------------------------------------------
# Carga
# --------------------------------------------------------------------------
def cargar_datos(ruta_csv: Path = RUTA_CSV,
                 eliminar_duplicados: bool = True) -> pd.DataFrame:
    """Carga PhiUSIIL desde el CSV local o, si no existe, desde UCI.

    Agrega la columna `es_phishing` (1 = phishing). En el dataset original
    la columna `label` vale 1 para legítima y 0 para phishing, al revés de
    lo intuitivo; por eso trabajamos con `es_phishing`.
    """
    ruta_csv = Path(ruta_csv)
    if ruta_csv.exists():
        df = pd.read_csv(ruta_csv, encoding="utf-8-sig")
    else:
        from ucimlrepo import fetch_ucirepo

        print("CSV local no encontrado: descargando desde UCI (id=967)...")
        repo = fetch_ucirepo(id=967)
        df = pd.concat([repo.data.features, repo.data.targets], axis=1)
        ruta_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ruta_csv, index=False)
        print(f"Guardado en {ruta_csv}")

    # La pauta de UCI indica que FILENAME se puede ignorar
    df = df.drop(columns=["FILENAME"], errors="ignore")
    # URLs repetidas (todas con la misma etiqueta): dejamos la primera.
    # Se conserva el índice original para que la división guardada calce.
    if eliminar_duplicados:
        df = df.drop_duplicates(subset="URL", keep="first")
    df[OBJETIVO] = (df["label"] == 0).astype(int)
    df["dominio_base"] = dominio_base(df["Domain"])
    return df


# --------------------------------------------------------------------------
# Normalización de texto
# --------------------------------------------------------------------------
def normalizar_url(urls: pd.Series) -> pd.Series:
    """Quita el esquema (http/https), el 'www.' inicial y la '/' final.

    Sirve para que un modelo de texto no aprenda atajos como
    "empieza con https://www. => legítima".
    """
    return (
        urls.str.lower()
        .str.replace(r"^https?://", "", regex=True)
        .str.replace(r"^www\.", "", regex=True)
        .str.rstrip("/")
    )


def dominio_base(dominios: pd.Series) -> pd.Series:
    """Dominio en minúsculas y sin 'www.' (ej. 'www.uni-mainz.de' -> 'uni-mainz.de')."""
    return dominios.str.lower().str.replace(r"^www\.", "", regex=True)


# --------------------------------------------------------------------------
# División train / val / test
# --------------------------------------------------------------------------
def crear_division(df: pd.DataFrame, prop_val: float = 0.15,
                   prop_test: float = 0.15, semilla: int = SEMILLA) -> pd.Series:
    """Divide en train/val/test agrupando por dominio.

    Todas las URLs de un mismo dominio quedan en el mismo conjunto. Así se
    evita que el modelo vea un dominio en entrenamiento y lo "reconozca" en
    test, lo que inflaría las métricas.
    """
    grupos = df["dominio_base"].values
    idx = np.arange(len(df))

    gss = GroupShuffleSplit(n_splits=1, test_size=prop_test, random_state=semilla)
    idx_resto, idx_test = next(gss.split(idx, groups=grupos))

    prop_val_rel = prop_val / (1 - prop_test)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=prop_val_rel, random_state=semilla)
    sub_train, sub_val = next(gss2.split(idx_resto, groups=grupos[idx_resto]))

    division = pd.Series("train", index=df.index, name="conjunto")
    division.iloc[idx_resto[sub_val]] = "val"
    division.iloc[idx_test] = "test"
    return division


def guardar_division(df: pd.DataFrame, division: pd.Series,
                     ruta: Path = RUTA_DIVISION) -> None:
    """Guarda a qué conjunto pertenece cada URL.

    Se guarda por URL (y no por número de fila) para que funcione aunque el
    CSV descargado tenga las filas en otro orden.
    """
    pd.DataFrame({"URL": df["URL"].values, "conjunto": division.values}) \
        .to_csv(ruta, index=False)


def cargar_division(ruta: Path = RUTA_DIVISION) -> pd.Series:
    """Lee la división guardada por 01_eda.ipynb (índice = URL)."""
    return pd.read_csv(ruta).set_index("URL")["conjunto"]


def cargar_conjuntos():
    """Devuelve (df, train, val, test) usando la división guardada."""
    df = cargar_datos()
    division = cargar_division()
    df["conjunto"] = df["URL"].map(division)
    if df["conjunto"].isna().any():
        raise ValueError("Hay URLs sin conjunto asignado: vuelva a ejecutar 01_eda.ipynb")
    return (df,
            df[df.conjunto == "train"],
            df[df.conjunto == "val"],
            df[df.conjunto == "test"])
