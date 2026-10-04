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
│   ├── Proyecto_Parte1_ML_Clasico.ipynb               Notebook principal
│   ├── Mejora1_Modelo_por_Oracion.ipynb               Modelo en dos etapas por oración
│   ├── Mejora2_Validacion_Cruzada_Repetida.ipynb      Decidir empates con 15 particiones
│   ├── Mejora3_Ajuste_Fino_Regresion_Logistica.ipynb  Grilla fina de C y elastic net
│   ├── Mejora4_Dos_Etapas_SVM.ipynb                   Dos etapas + SVM lineal y comparación justa de modelos
│   └── Exploracion_modelos_alternativos.ipynb         Otros modelos, votación y apilamiento
├── html/                  Versión HTML de cada notebook
├── models/                Modelos entrenados (.joblib), uno por modelo
├── submissions/           Predicciones listas para subir a Kaggle (id, answer), una por modelo
├── requirements.txt
└── README.md
```

**Convención para varios modelos:** cada modelo se identifica con un nombre (`NOMBRE_MODELO` en el notebook) y deja dos archivos con ese nombre: `models/<nombre>.joblib` y `submissions/submission_<nombre>.csv`. Así los envíos a Kaggle no se pisan entre sí.

Para Bloque Neón se sube un único modelo: el que corresponda al mejor envío de Kaggle del grupo.

## Resumen de los modelos

Todos usan la misma partición 80/20 (semilla 42). La columna de 15 particiones viene de la comparación justa del notebook `Mejora4_Dos_Etapas_SVM.ipynb` (3 repeticiones de validación cruzada de 5 particiones sobre las 12.000 reseñas), que es la estimación más confiable. La métrica de Kaggle es la exactitud y el puntaje público se calcula sobre solo 900 reseñas (el 30% de `eval.csv`), con un error de cerca de ±1 punto; normalmente las 2.100 restantes son las que definen el ranking final.

| Modelo (nombre de archivo) | Exactitud test | F1-macro test | Exactitud 15 particiones | Kaggle público | Predicciones distintas vs actual |
|:---|:---:|:---:|:---:|:---:|:---:|
| `dos_etapas_svm` (Mejora 4) | 0,8988 | 0,9036 | **0,9009** | 0,89888 | 127 |
| `dos_etapas_por_oracion` (Mejora 1) | 0,8971 | 0,9021 | 0,8997 | 0,90222 | 120 |
| `exploracion_apilamiento` (exploración) | 0,9000 | 0,9047 | 0,8991 | 0,90000 | 112 |
| `regresion_logistica_ajuste_fino` (Mejora 3) | 0,8975 | 0,9019 | empate con el actual (Mejora 3) | 0,90666 | 30 |
| `regresion_logistica_bloques` (modelo actual) | 0,8946 | 0,8993 | 0,8966 | 0,90222 | 0 |
| `regresion_logistica_cv_repetida` (Mejora 2) | 0,8946 | 0,8993 | igual al actual | 0,90222 | 0 (idéntico) |

Los tres primeros mejoran al modelo actual entre 0,25 y 0,42 puntos de forma consistente, pero entre ellos las diferencias están dentro del ruido. En el puntaje público, contado en aciertos de 900, el ajuste fino tiene 4 más que el modelo actual, el apilamiento 2 menos y la Mejora 4 3 menos: diferencias dentro del ruido que no permiten distinguir los modelos (ver la sección 10 del notebook de la Mejora 4).

## Cómo replicar los resultados

1. Crear un entorno con Python 3.9 e instalar las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

2. Abrir el notebook que se quiera replicar (por ejemplo `notebooks/Proyecto_Parte1_ML_Clasico.ipynb`) y ejecutarlo completo (`Restart & Run All`). El notebook debe ejecutarse desde la carpeta `notebooks/`, porque lee los datos de `../data`.
3. La primera celda descarga las stop words de NLTK solo si no están en el computador, por lo que puede requerir conexión a internet esa única vez.
4. Al terminar se regeneran `models/regresion_logistica_bloques.joblib` y `submissions/submission_regresion_logistica_bloques.csv`.

Todas las fuentes de aleatoriedad usan `RANDOM_STATE = 42` (partición train/test, validación cruzada y modelos), por lo que las cifras del notebook se reproducen exactamente con las versiones de `requirements.txt`. El notebook principal tarda alrededor de 12 minutos por las búsquedas de hiperparámetros; los de mejora tardan entre 4 y 15 minutos.

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
