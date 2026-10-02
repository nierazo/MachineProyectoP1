# Proyecto - Parte 1: Machine Learning clásico

Clasificación de reseñas de productos en español (`positivo`, `negativo`, `neutral`) usando únicamente modelos clásicos de scikit-learn y representaciones de texto vistas en el curso (preprocesamiento, TF-IDF y n-gramas).

## Archivos

| Archivo | Descripción |
|---|---|
| `Proyecto_Parte1_ML_Clasico.ipynb` | Notebook con todo el desarrollo (exploración, preprocesamiento, modelos, evaluación y predicciones), con las celdas ejecutadas. |
| `Proyecto_Parte1_ML_Clasico.html` | Versión en HTML del notebook ejecutado. |
| `modelo_final.joblib` | Único modelo entregado: pipeline completo (preprocesamiento + TF-IDF + clasificador) entrenado con todos los datos etiquetados. |
| `submission.csv` | Predicciones sobre `eval.csv` en el formato de `sample_submission.csv` (`id`, `answer`). |
| `requirements.txt` | Versiones de las librerías con las que se ejecutó. |
| `train.csv`, `eval.csv`, `sample_submission.csv` | Datos de la competencia (sin modificar). |

## Cómo replicar los resultados

1. Crear un entorno con Python 3.9 e instalar las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

2. Dejar el notebook en la misma carpeta que `train.csv`, `eval.csv` y `sample_submission.csv`.
3. Ejecutar el notebook completo (`Kernel > Restart & Run All`). La primera celda descarga las stop words de NLTK, por lo que requiere conexión a internet esa única vez.
4. Al terminar se regeneran `submission.csv` y `modelo_final.joblib`.

Todas las fuentes de aleatoriedad usan `RANDOM_STATE = 42` (partición train/test, validación cruzada y modelos), por lo que las cifras del notebook se reproducen exactamente con las versiones de `requirements.txt`. El notebook completo tarda alrededor de 15 minutos por las búsquedas de hiperparámetros.

## Usar el modelo guardado fuera del notebook

El pipeline guardado referencia las funciones de preprocesamiento definidas en el notebook (secciones 5 y 8). Para cargarlo en otra sesión hay que ejecutar primero las celdas que definen esas funciones y después:

```python
import joblib
import pandas as pd

modelo = joblib.load('modelo_final.joblib')
evaluacion = pd.read_csv('eval.csv')
predicciones = modelo.predict(evaluacion[['text']])
```
