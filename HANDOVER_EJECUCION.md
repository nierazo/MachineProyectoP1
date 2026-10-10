# Handover: ejecutar el notebook final en otro computador

## Qué hay que hacer

Ejecutar completo (`Restart & Run All`) el notebook:

```
notebooks/Proyecto_Parte1_Final.ipynb
```

Es el notebook de entrega. Ya está terminado y revisado; solo falta correrlo de principio a fin para que queden las salidas (tablas, gráficas, métricas) y para que regenere:

- `models/dos_etapas_nblr_robusto.joblib`
- `submissions/submission_dos_etapas_nblr_robusto.csv`

**No hay que escribir ni cambiar nada del contenido**, solo ejecutarlo.

## Por qué se está moviendo a otro computador

El notebook entrena varias veces un modelo en dos etapas sobre las 12.000 reseñas (una vez solo, con los arreglos de robustez, y 30 veces más para la comparación de 15 particiones de la sección 9, que es la parte más pesada). En el computador original cada entrenamiento completo toma 2–3 minutos, así que el notebook completo tarda entre 35 y 60 minutos. Si el otro computador es más rápido, debería tardar menos.

## Cómo correrlo

### Paso 1: entorno

Crear un entorno con Python 3.9 e instalar las dependencias (están en la raíz del proyecto):

```bash
pip install -r requirements.txt
```

Paquetes clave y versión con la que se probó: `scikit-learn==1.6.1`, `pandas==2.3.3`, `numpy==2.0.2`, `nltk==3.9.2`, `joblib==1.5.3`. El *stemmer* en español de NLTK (`SnowballStemmer`) no necesita descargar datos adicionales, así que no hace falta conexión a internet para esa parte.

### Paso 2: estructura de carpetas

Mantener la estructura del proyecto tal cual está: el notebook debe quedar en `notebooks/`, los datos en `data/` (ya están ahí: `train.csv`, `eval.csv`, `sample_submission.csv`), y los resultados se escriben en `models/` y `submissions/` (ya existen, no hay que crearlas).

### Paso 3: ejecutar

**Opción A — desde Jupyter/JupyterLab/VS Code:** abrir `notebooks/Proyecto_Parte1_Final.ipynb` y usar "Restart Kernel and Run All Cells".

**Opción B — desde la terminal** (más cómodo para dejarlo corriendo sin supervisión), parado en la carpeta `notebooks/`:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=5400 Proyecto_Parte1_Final.ipynb
```

El `--timeout=5400` (90 minutos por celda) es solo un margen de seguridad para que no corte la celda de la comparación de 15 particiones si esa máquina es más lenta de lo esperado; en la práctica ninguna celda debería tardar más de 20-25 minutos.

### Qué esperar mientras corre

- Es normal que no haya salida en la consola hasta que termina (o falla) — `nbconvert` no imprime progreso por celda.
- La celda más pesada es la de la sección "Comparación justa con 15 particiones": reentrena el modelo en dos etapas 30 veces. Si se quiere confirmar que no está trabado, revisar que el proceso de Python esté usando CPU alta (no 0%) en el administrador de tareas.
- Al terminar sin errores, el notebook queda con todas las celdas numeradas y sus salidas (tablas, gráficas) visibles, y se habrán regenerado los dos archivos mencionados arriba.

### Si algo fallara

Si una celda da un error real (no un timeout), lo más probable es un problema de versiones de librerías: confirmar que se usó `requirements.txt` del proyecto y no otro entorno. Si es un timeout, subir el valor de `--timeout` y volver a correr desde cero (`--inplace` sobreescribe el archivo, así que no hace falta borrar nada antes).

## Verificación rápida al terminar

```python
import pandas as pd
df = pd.read_csv('submissions/submission_dos_etapas_nblr_robusto.csv')
print(df.shape)              # debe ser (3000, 2)
print(df['answer'].value_counts(normalize=True).round(3))
```

Las proporciones esperadas son aproximadamente 35-36% `negativo`, 27-28% `neutral` y 36% `positivo`.
