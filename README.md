# Detección de URLs de phishing con Machine Learning y Deep Learning

Proyecto semestral · IAMLDL 2027-1 · Universidad del Bío-Bío

Este proyecto compara modelos de Machine Learning (sobre variables tabulares) y de Deep Learning (sobre el texto de la URL) para clasificar URLs como **phishing** o **legítimas**, y mide cuánto de su desempeño se debe a **atajos** del dataset.

## Resultado principal

| Modelo (test, 35.246 URLs) | ¿Usa atajos? | F1 phishing |
|---|---|---|
| Random Forest, todas las variables | Sí | 1,000 |
| CNN de caracteres, URL original | Sí | 0,997 |
| TF-IDF + regresión logística, solo dominio | No | 0,830 |
| CNN de caracteres, solo dominio | No | 0,825 |
| BiGRU, solo dominio | No | 0,813 |
| XGBoost, variables léxicas propias del dominio | No | 0,777 |

Todas las URLs legítimas del dataset empiezan con `https://www.`, no tienen ruta y tienen `URLSimilarityIndex = 100`. Con esos atajos cualquier modelo llega a ~100 %, pero el modelo de URL original marca como phishing cualquier URL legítima con ruta (por ejemplo `https://github.com/54isaac09`). Sin atajos, mirando solo el dominio, el mejor F1 es ~0,83.

## Dataset

**PhiUSIIL Phishing URL Dataset** (UCI, id 967): 235.795 URLs (134.850 legítimas y 100.945 de phishing) con 54 variables.

> A. Prasad y S. Chandra, "PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning," *Computers & Security*, vol. 136, Art. no. 103545, ene. 2024, doi: 10.1016/j.cose.2023.103545.

El CSV no se sube al repositorio (54 MB). La primera vez que se ejecuta el código, se descarga automáticamente desde UCI con `ucimlrepo` y se guarda en `data/`.

**Importante:** en el dataset original `label = 1` es legítima y `label = 0` es phishing. El código crea la columna `es_phishing` (1 = phishing) y se trabaja siempre con ella.

## Cómo reproducir

```bash
git clone https://github.com/54isaac09/Machine-Learning-phishing-url-detection.git
cd Machine-Learning-phishing-url-detection
python -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Ejecutar los notebooks en orden:

| Notebook | Etapa | Contenido | Tiempo aprox. (CPU) |
|---|---|---|---|
| `notebooks/01_eda.ipynb` | E-01, E-02 | Carga, calidad de datos, atajos, división train/val/test | < 1 min |
| `notebooks/02_modelos_tabulares.ipynb` | E-04, E-05 | Regresión logística, Random Forest, XGBoost y MLP en 4 escenarios; ajuste; SHAP | ~3 min |
| `notebooks/03_modelos_texto.ipynb` | E-04, E-05 | TF-IDF, CNN de caracteres y BiGRU sobre URL y dominio | 15-30 min |
| `notebooks/04_evaluacion_final.ipynb` | E-06 | Evaluación en test, matrices de confusión, ROC, umbral, errores, sesgo residual | ~1 min |

Los modelos entrenados ya están en `models/`, así que el notebook 04 y la demo funcionan sin volver a entrenar.

La división train/val/test (70/15/15, agrupada por dominio, semilla 42) queda en `data/division.csv` y la usan todos los notebooks.

## Demo

La demo analiza el texto de la URL **sin visitarla**.

```bash
python demo.py                                          # modo interactivo
python demo.py "http://secure-login-verify-account.xyz/signin"
python demo.py --comparar "https://github.com/54isaac09"  # compara con el modelo que aprendió los atajos
python demo.py --umbral 0.3 "ejemplo.com"               # umbral más bajo: detecta más phishing
```

## Estructura

```
├── README.md
├── requirements.txt
├── demo.py                   # clasifica una URL desde la terminal
├── data/
│   └── division.csv          # conjunto (train/val/test) de cada URL
├── notebooks/                # 01 a 04, en orden
├── src/
│   ├── datos.py              # carga, normalización y división
│   ├── caracteristicas.py    # variables léxicas propias del dominio
│   ├── texto.py              # extracción del dominio y codificación de caracteres
│   └── evaluacion.py         # métricas
├── models/                   # modelos entrenados (.joblib y .keras)
├── reports/
│   ├── figures/              # figuras para el informe
│   └── resultados/           # tablas de resultados (CSV y JSON)
└── informe/                  # informe en formato IEEE (LaTeX y PDF)
```

## Integrantes

- Nombre Apellido
- Nombre Apellido
