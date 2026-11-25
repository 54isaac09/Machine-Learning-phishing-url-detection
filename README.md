# Detección de URLs de phishing con Machine Learning y Deep Learning

Proyecto semestral · IAMLDL 2027-1 · Universidad del Bío-Bío

Este proyecto compara modelos de Machine Learning (sobre variables tabulares) y de Deep Learning (sobre el texto de la URL) para clasificar URLs como **phishing** o **legítimas**. También analiza cuánto del desempeño se debe a **atajos** del dataset.

## Dataset

**PhiUSIIL Phishing URL Dataset** (UCI, id 967): 235.795 URLs (134.850 legítimas y 100.945 de phishing) con 54 variables.

> A. Prasad y S. Chandra, "PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning," *Computers & Security*, vol. 136, Art. no. 103545, ene. 2024, doi: 10.1016/j.cose.2023.103545.

El CSV no se sube al repositorio (54 MB). La primera vez que se ejecuta el código, se descarga automáticamente desde UCI con `ucimlrepo` y se guarda en `data/`.

**Importante:** en el dataset original `label = 1` es legítima y `label = 0` es phishing. El código crea la columna `es_phishing` (1 = phishing) y se trabaja siempre con ella.

## Cómo reproducir

```bash
git clone <url-del-repositorio>
cd phishing-url-detection
python -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Ejecutar los notebooks en orden:

| Notebook | Etapa | Contenido |
|---|---|---|
| `notebooks/01_eda.ipynb` | E-01, E-02 | Carga, calidad de datos, atajos, división train/val/test |
| `notebooks/02_modelos_tabulares.ipynb` | E-04 a E-06 | *(pendiente)* Regresión logística, Random Forest, XGBoost, MLP |
| `notebooks/03_modelos_texto.ipynb` | E-04 a E-06 | *(pendiente)* TF-IDF, CNN de caracteres, LSTM/BiGRU |
| `demo.py` | Presentación | *(pendiente)* Clasifica una URL ingresada |

La división train/val/test (70/15/15, agrupada por dominio, semilla 42) queda en `data/division.csv` y la usan todos los notebooks.

## Estructura

```
phishing-url-detection/
├── README.md
├── requirements.txt
├── data/
│   └── division.csv          # conjunto (train/val/test) de cada URL
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   └── datos.py              # carga, normalización y división
├── models/                   # modelos entrenados
└── reports/figures/          # figuras para el informe
```

## Hallazgos principales del análisis exploratorio

- Todas las URLs legítimas empiezan con `https://www.`, no tienen ruta y tienen `URLSimilarityIndex = 100`. Son atajos que vienen de cómo se construyó el dataset.
- Un modelo simple de texto obtiene 99,7 % de accuracy con la URL original, pero cae a ~90 % al normalizarla y a ~87 % usando solo el dominio.
- Por eso los modelos se evalúan con y sin esos atajos.

## Integrantes

- Nombre Apellido
- Nombre Apellido
