# Proyecto - Parte 1: Machine Learning clásico

Clasificación de reseñas de productos en español (`positivo`, `negativo`, `neutral`) usando únicamente modelos clásicos de scikit-learn y representaciones de texto vistas en el curso (preprocesamiento, TF-IDF y n-gramas).

## Estructura del proyecto

```
MachineProyectoP1/
├── data/                  Datos de la competencia (sin modificar)
│   ├── train.csv
│   ├── eval.csv
│   └── sample_submission.csv
├── notebooks/             Notebooks con las celdas ejecutadas
│   └── Proyecto_Parte1_ML_Clasico.ipynb
├── html/                  Versión HTML de cada notebook
│   └── Proyecto_Parte1_ML_Clasico.html
├── models/                Modelos entrenados (.joblib)
│   └── regresion_logistica_bloques.joblib
├── submissions/           Predicciones listas para subir a Kaggle (id, answer)
│   └── submission_regresion_logistica_bloques.csv
├── requirements.txt
└── README.md
```

**Convención para varios modelos:** cada modelo se identifica con un nombre (`NOMBRE_MODELO` en el notebook) y deja dos archivos con ese nombre: `models/<nombre>.joblib` y `submissions/submission_<nombre>.csv`. Así los envíos a Kaggle no se pisan entre sí.

Para Bloque Neón se sube un único modelo: el que corresponda al mejor envío de Kaggle del grupo.

## Cómo replicar los resultados

1. Crear un entorno con Python 3.9 e instalar las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

2. Abrir `notebooks/Proyecto_Parte1_ML_Clasico.ipynb` y ejecutarlo completo (`Restart & Run All`). El notebook debe ejecutarse desde la carpeta `notebooks/`, porque lee los datos de `../data`.
3. La primera celda descarga las stop words de NLTK solo si no están en el computador, por lo que puede requerir conexión a internet esa única vez.
4. Al terminar se regeneran `models/regresion_logistica_bloques.joblib` y `submissions/submission_regresion_logistica_bloques.csv`.

Todas las fuentes de aleatoriedad usan `RANDOM_STATE = 42` (partición train/test, validación cruzada y modelos), por lo que las cifras del notebook se reproducen exactamente con las versiones de `requirements.txt`. El notebook completo tarda alrededor de 12 minutos por las búsquedas de hiperparámetros.

Las predicciones del modelo final también se verificaron con Python 3.11.5 y scikit-learn 1.5.2: las 3.000 predicciones sobre `eval.csv` son idénticas. Las únicas diferencias entre entornos están en la cuarta cifra decimal de Random Forest y del árbol de decisión, que usan azar interno.

## Usar el modelo guardado fuera del notebook

El pipeline guardado referencia las funciones de preprocesamiento definidas en el notebook (secciones 5, 7 y 8). Para cargarlo en otra sesión hay que ejecutar primero las celdas que definen esas funciones y después:

```python
import joblib
import pandas as pd

modelo = joblib.load('../models/regresion_logistica_bloques.joblib')
evaluacion = pd.read_csv('../data/eval.csv')
predicciones = modelo.predict(evaluacion[['text']])
```
