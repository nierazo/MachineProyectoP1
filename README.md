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
│   ├── Proyecto_Parte1_Final.ipynb                    Notebook de entrega: todo el proceso y el modelo final
│   ├── Proyecto_Parte1_ML_Clasico.ipynb               Notebook principal (primera versión del modelo)
│   ├── Mejora1_Modelo_por_Oracion.ipynb               Modelo en dos etapas por oración
│   ├── Mejora2_Validacion_Cruzada_Repetida.ipynb      Decidir empates con 15 particiones
│   ├── Mejora3_Ajuste_Fino_Regresion_Logistica.ipynb  Grilla fina de C y elastic net
│   ├── Mejora4_Dos_Etapas_SVM.ipynb                   Dos etapas + SVM lineal y comparación justa de modelos
│   ├── Mejora5_Marcado_Negacion.ipynb                 Marcado del alcance de la negación ("no X" -> "X_neg")
│   ├── Mejora6_Ensamble_Voto_Suave.ipynb              Voto suave de los 5 modelos candidatos
│   ├── Mejora7_NBLR_Dos_Etapas.ipynb                  Corrección de la cláusula final, NB-LR y dos etapas + NB-LR
│   ├── Mejora8_Ensambles_Train_Val_Test.ipynb         Ensambles con partición entrenamiento/validación/prueba
│   ├── Diagnostico_Modelo_Final.ipynb                 Sobreajuste, fugas, sesgo y robustez del modelo final (Mejora 9)
│   ├── Mejora9_Robustez_Opinion.ipynb                 Aumento de datos y última oración con opinión (robustez)
│   ├── Exploracion_modelos_alternativos.ipynb         Otros modelos, votación y apilamiento
│   ├── PruebaF_NBLR_Dos_Etapas.ipynb                  Prueba: n-gramas de caracteres en NB-LR (empata con la Mejora 7)
│   ├── samuel_Mejora9_KNN_Compuerta_Tres_Regresiones.ipynb  Tres regresiones NB-LR con compuerta KNN (no mejora)
│   ├── samuel_Mejora10_Techo_de_Exactitud.ipynb             Análisis del error y techo de exactitud (sin modelo nuevo)
│   └── samuel_Mejora11_Ortografia_y_Ensamble.ipynb          Corrección ortográfica y ensambles (no cumplen la regla)
├── html/                  Versión HTML de cada notebook
├── models/                Modelos entrenados (.joblib), uno por modelo
├── submissions/           Predicciones listas para subir a Kaggle (id, answer), una por modelo
├── requirements.txt
├── HANDOVER_EJECUCION.md  Instrucciones para ejecutar el notebook de entrega en otro computador
├── Resumen_experimentos.md
└── README.md
```

**Convención para varios modelos:** cada modelo se identifica con un nombre (`NOMBRE_MODELO` en el notebook) y deja dos archivos con ese nombre: `models/<nombre>.joblib` y `submissions/submission_<nombre>.csv`. Así los envíos a Kaggle no se pisan entre sí.

Para Bloque Neón se sube un único modelo: el que corresponda al mejor envío de Kaggle del grupo. Hoy es `dos_etapas_nblr_robusto` (Mejora 9), que además es el mejor en la comparación de 15 particiones.

## Resumen de los modelos

Todos usan la misma partición 80/20 (semilla 42), salvo la Mejora 8, que usa 60/20/20. La columna de 15 particiones viene de la comparación justa de los notebooks de las Mejoras 4 y 7: 3 repeticiones de validación cruzada de 5 particiones sobre las 12.000 reseñas, que es la estimación más confiable. La métrica de Kaggle es la exactitud. El puntaje público se calcula sobre solo 900 reseñas (el 30% de `eval.csv`), con un error de cerca de ±1 punto, y normalmente las 2.100 restantes son las que definen el ranking final.

| Modelo (nombre de archivo) | Exactitud test | F1-macro test | Exactitud 15 particiones | Kaggle público | Predicciones distintas vs actual |
|:---|:---:|:---:|:---:|:---:|:---:|
| `dos_etapas_nblr_robusto` (Mejora 9) | 0,9025 | no reportado | **0,9081** | **0,91777** | 191 |
| `regresion_logistica_robusta` (Mejora 9) | 0,9033 | no reportado | 0,9046 | 0,91666 | 157 |
| `samuel_ensamble_ortografia` (Mejora 11 de Samuel) | no reportado | no reportado | 0,9035 | sin registrar | no calculado |
| `dos_etapas_nblr` (Mejora 7) | 0,8988 | 0,9037 | 0,9023 | 0,90777 | 145 |
| `dos_etapas_svm` (Mejora 4) | 0,8988 | 0,9036 | 0,9009 | 0,89888 | 127 |
| `nb_regresion_logistica` (Mejora 7) | 0,8983 | 0,9029 | 0,9004 | sin registrar | 185 |
| `dos_etapas_por_oracion` (Mejora 1) | 0,8971 | 0,9021 | 0,8997 | 0,90222 | 120 |
| `exploracion_apilamiento` (exploración) | 0,9000 | 0,9047 | 0,8991 | 0,90000 | 112 |
| `regresion_logistica_corregida` (Mejora 7) | 0,8954 | 0,9000 | 0,8988 | sin registrar | 20 |
| `regresion_logistica_ajuste_fino` (Mejora 3) | 0,8975 | 0,9019 | empate con el actual (Mejora 3) | 0,90666 | 30 |
| `regresion_logistica_bloques` (modelo actual) | 0,8946 | 0,8993 | 0,8966 | 0,90222 | 0 |
| `regresion_logistica_cv_repetida` (Mejora 2) | 0,8946 | 0,8993 | igual al actual | 0,90222 | 0 (idéntico) |
| `regresion_logistica_negacion` (Mejora 5) | 0,8971 | 0,9014 | empate con el actual (Mejora 5) | 0,90333 | 55 |
| `ensamble_voto_suave` (Mejora 6) | 0,9004 | 0,9051 | ver nota (Mejora 6, bootstrap) | sin registrar | 72 |
| `samuel_knn_compuerta_tres_regresiones` (Mejora 9 de Samuel) | 0,9029 (partición 60/20/20) | no reportado | no evaluada | sin registrar | no calculado |
| `votacion_suave_mejores` (Mejora 8) | 0,9025 (otra partición) | 0,9070 (otra partición) | no evaluada | sin registrar | 77 |

**El mejor modelo en la comparación de 15 particiones es `dos_etapas_nblr_robusto` (Mejora 9):** 0,9081, +0,59 puntos sobre la Mejora 7, ganando en 14 de 15. También tiene el mejor puntaje público del grupo (0,91777 = 826/900) y es robusto a frases sin opinión al final de la reseña (ver la nota de la Mejora 9).

**Antes de la Mejora 9, el mejor modelo era `dos_etapas_nblr` (Mejora 7):** tiene 0,9023 en la comparación de 15 particiones (+0,56 puntos sobre el modelo actual, ganando en 14 de 15) y el mejor puntaje público del grupo hasta entonces (0,90777 = 817/900). Su ventaja sobre las demás alternativas fuertes (Mejora 4, NB-LR, votación de la Mejora 8) es de décimas y está dentro del ruido. En el puntaje público, contado en aciertos de 900, las diferencias entre envíos son de pocas reseñas (entre 809 y 817), dentro del error de ±1 punto (ver la sección 10 del notebook de la Mejora 4).

La Mejora 5 marca el alcance de la negación ("no funciona bien" -> "no funciona_neg bien_neg") en las dos oraciones finales de la reseña. En la comparación de 15 particiones gana en 11 de 15 (el mínimo para contar como mejora es 12), así que también es un empate con el modelo actual. Su puntaje público (0,90333 = 813/900) confirma esa lectura: 1 acierto más que el modelo actual (812/900), una diferencia que cabe dentro del error de ±1 punto del puntaje público.

La Mejora 6 promedia las probabilidades (voto suave, `VotingClassifier`) de cinco modelos (actual, ajuste fino, apilamiento, dos etapas y dos etapas + SVM). Como dos de los candidatos tardan minutos en entrenarse, se comparó con una prueba de bootstrap pareado sobre el conjunto de prueba en vez de las 15 particiones: el ensamble mejora de forma clara al modelo actual (gana en 97,9 % de los remuestreos) y probablemente al modelo en dos etapas (92,3 %) y a la SVM de la Mejora 4 (74,1 %), pero frente al ajuste fino es solo un indicio (84,2 %, no concluyente) y frente al apilamiento es un empate (54,5 %), porque el apilamiento ya combina modelos por su cuenta.

La Mejora 7 hace tres cosas. Primero corrige un error de la representación del notebook principal: cortar la cláusula final en "al final de cuentas" la dejaba vacía en cerca del 3,5% de las reseñas; corregirlo mejora al modelo actual en 0,21 puntos, en las 15 de 15 particiones. Después prueba NB-LR (Wang y Manning, 2012), una regresión logística sobre la bolsa de palabras ponderada con las razones de Naive Bayes, que mejora 0,37 puntos. Por último combina NB-LR con el modelo en dos etapas, que es el mejor modelo del proyecto. Desde esta mejora los modelos se eligen por exactitud, que es la métrica de Kaggle.

La Mejora 8 responde a los comentarios del profesor: usa una partición en entrenamiento, validación y prueba (60/20/20), mide el sesgo de los modelos y revisa los ensambles siguiendo la práctica. La votación suave de los 2 mejores modelos (regresión logística y dos etapas + NB-LR) es el mejor ensamble: 0,8996 en validación y 0,9025 en prueba, frente a 0,8967 y 0,8946 de la regresión logística. El stacking la iguala, y Random Forest y Gradient Boosting rinden mucho peor sobre texto. Su exactitud de prueba no es comparable con la del resto de la tabla, porque se mide sobre otra partición. Su ventaja sobre dos etapas + NB-LR es pequeña, y su envío cambia solo 69 predicciones respecto al de ese modelo.

La Mejora 9 corrige la principal debilidad que encontró la primera versión del notebook de diagnóstico, hecha con la Mejora 7: con una frase sin opinión al final de la reseña (por ejemplo "Viene en color negro."), los modelos pierden entre 20 y 40 puntos de exactitud, porque le dan mucho peso a la última oración. Prueba dos arreglos: (A) aumento de datos, que agrega al final de la mitad de las reseñas de entrenamiento una oración de una reseña neutral sin cambiar su etiqueta, y (B) usar la última oración con opinión, elegida por un detector opinión/hecho (TF-IDF + regresión logística entrenada solo con `train.csv`). Con los dos arreglos, la exactitud se mantiene cerca de 0,90 con frases neutras al final y además sube la exactitud normal: la Mejora 7 + A + B llega a 0,9081 y la regresión logística + A + B a 0,9046 (+0,58, ganando en 14 de 15). La Mejora 7 + A + B supera a la regresión logística + A + B en 13 de 15 particiones, apenas por encima del mínimo de 12, y esa diferencia depende de la semilla del aumento de datos, así que es la conclusión menos segura. Frente al envío de la Mejora 7, `dos_etapas_nblr_robusto` cambia 113 de las 3.000 predicciones, por lo que la mejora esperada en el puntaje público era de unas 5 reseñas de 900, menos que el error de ±1 punto. En Kaggle subió de 817 a 826 aciertos (0,91777), y la regresión logística + A + B sacó 825 (0,91666). La diferencia de 1 reseña entre los dos es ruido, y los dos puntajes quedan cerca de 1 error estándar por encima de lo que predice la comparación de 15 particiones (unos 817 y 814 aciertos), así que es probable que el puntaje final sea algo menor que el público. El notebook de diagnóstico, actualizado al modelo final, lo confirma con validación cruzada: con una frase neutra al final la exactitud baja solo entre 0,1 y 0,2 puntos. También muestra que el modelo final sobreajusta menos que el de la Mejora 7 (7,4 contra 8,8 puntos de diferencia entre entrenamiento y validación) y, con una prueba de permutación, que el aumento de datos y el detector no filtran información de validación.

Los notebooks que empiezan con `samuel_` son las Mejoras 9, 10 y 11 de Samuel, que se hicieron en paralelo a la Mejora 9 de robustez; el prefijo evita confundirlas con ella. Ninguno supera al modelo final: la compuerta KNN empata con las regresiones sin compuerta (0,9029 frente a 0,9038 en prueba), el análisis del techo estima un máximo realista cercano a 0,905 a 0,91 porque el 12% de las reseñas termina en un cierre de duda que es azar respecto del texto, y la corrección ortográfica y los ensambles suben a lo sumo 0,1 puntos sobre la Mejora 7 (0,9035), sin cumplir la regla de 12 de 15 particiones.

Un resumen de todos los experimentos, incluidos los que no mejoraron, está en `Resumen_experimentos.md`.

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
