# Resumen de experimentos - Proyecto Parte 1

Clasificación de reseñas en español (`positivo`, `negativo`, `neutral`) con 12.000 reseñas etiquetadas (`train.csv`) y 3.000 para la competencia (`eval.csv`). La métrica de Kaggle es la exactitud.

## 1. Protocolo de evaluación

- **Partición 80/20 estratificada** (semilla 42): el 20% de prueba no se usa para elegir nada. La Mejora 8 usa además una partición **60/20/20** (entrenamiento, validación y prueba), como pidió el profesor.
- **Comparación justa con 15 particiones:** 3 repeticiones de validación cruzada estratificada de 5 particiones (semillas 1, 2 y 3) sobre las 12.000 reseñas. Todos los modelos se evalúan en las mismas particiones y se comparan partición por partición.
- **Regla de decisión:** una alternativa mejora si la diferencia media supera 2 errores estándar y gana en al menos 12 de las 15 particiones. Si no, es un empate y se mantiene la opción más simple.
- **Puntaje público de Kaggle:** se calcula con solo 900 reseñas (el 30% de `eval.csv`), con un error de ±1 punto. Se usa para confirmar, no para elegir.

## 2. Representación del texto (notebook principal)

| Paso | Resultado |
|:---|:---|
| Línea base: TF-IDF (1-2 gramas) del texto completo + regresión logística | Exactitud en prueba de 0,728. `neutral` casi perfecto; `positivo` y `negativo` se confunden (F1 ≈ 0,61). |
| Análisis de errores: la etiqueta la define la opinión del final de la reseña | F1-macro con solo la primera oración: 0,517; solo la última: 0,860; texto completo: 0,744. |
| Representación por bloques: texto completo + última oración + cláusula después del último conector de contraste | F1-macro en validación cruzada de 0,744 → 0,886 → 0,894. Es la mejora más grande del proyecto. |
| Preprocesamiento | Quitar tildes y signos ayuda levemente; el stemming no cambia el resultado; las stop words de NLTK empeoran (eliminan "no", "pero", "muy", "sin"). |
| Algoritmos del curso sobre la misma representación (exactitud en prueba) | Regresión logística 0,895, Naive Bayes 0,886, Random Forest 0,841, KNN 0,796, árbol de decisión 0,674. |

## 3. Mejoras (exactitud con 15 particiones)

| Notebook | Modelo | Exactitud 15 particiones | vs. modelo actual | Kaggle público |
|:---|:---|:---:|:---:|:---:|
| Mejora 9 | **Dos etapas + NB-LR + arreglos de robustez (A + B)** | **0,9081** | +0,59 vs. Mejora 7 (14/15) | **0,91777** (826/900) |
| Mejora 9 | Regresión logística + arreglos de robustez (A + B) | 0,9046 | +0,58 vs. cláusula corregida (14/15) | 0,91666 (825/900) |
| Mejora 7 | Dos etapas + NB-LR | 0,9023 | +0,56 (14/15) | 0,90777 (817/900) |
| Mejora 4 | Dos etapas + SVM lineal | 0,9009 | +0,42 (14/15) | 0,89888 (809/900) |
| Mejora 7 | NB-LR | 0,9004 | +0,37 (13/15) | sin dato |
| Mejora 1 | Dos etapas por oración | 0,8997 | +0,31 (12/15) | 0,90222 (812/900) |
| Exploración | Apilamiento (regresión logística, Naive Bayes, SVM) | 0,8991 | +0,25 (13/15) | 0,90000 (810/900) |
| Mejora 7 | Modelo actual con la cláusula corregida | 0,8988 | +0,21 (15/15) | sin dato |
| Mejora 3 | Regresión logística, ajuste fino (L1, C = 7) | empate con el actual | +0,07 F1 (8/15) | 0,90666 (816/900) |
| Mejora 5 | Marcado de negación | empate con el actual | 11/15 | 0,90333 (813/900) |
| Notebook principal | Modelo actual: regresión logística L1 (C = 10) por bloques | 0,8966 | — | 0,90222 (812/900) |
| Mejora 2 | Validación cruzada repetida para decidir empates | idéntico al actual | — | 0,90222 (812/900) |

Otros resultados:

- **Mejora 2:** con 15 particiones se confirmaron las decisiones del notebook principal. Pasar a unigramas empeora 1,16 puntos de F1, quitar el bloque de cláusula final empeora 0,67 y la lista ampliada de conectores empeora 0,31. Trigramas, stemming, stop words y conectores adicionales empatan. Una sola validación cruzada sobrestimaba el desempeño en unos 0,25 puntos.
- **Mejora 3:** elastic net (solver `saga`) quedó por debajo de L1 (F1 0,8973 contra 0,8998), tarda unas 30 veces más por ajuste y no converge con el límite de iteraciones usado.
- **Mejora 6 (voto suave de 5 modelos):** según un bootstrap pareado sobre la partición de prueba, mejora al modelo actual (97,9% de los remuestreos), pero frente al ajuste fino solo hay un indicio (84,2%) y frente al apilamiento es un empate (54,5%).
- **Mejora 7:** se encontró y corrigió un error de la representación. Con "al final de cuentas", la cláusula final quedaba vacía en cerca del 3,5% de las reseñas.
- **Mejora 8 (partición 60/20/20):** la votación suave de los 2 mejores modelos (regresión logística y dos etapas + NB-LR) obtuvo 0,8996 en validación y 0,9025 en prueba, contra 0,8967 y 0,8946 de la regresión logística. El stacking iguala a la regresión logística, y Random Forest (0,776) y Gradient Boosting (0,803) rinden mucho peor sobre texto. Los modelos lineales tienen 8 a 10 puntos de diferencia entre entrenamiento y validación, y la curva de aprendizaje muestra que más datos del mismo tipo ayudarían poco. Con la comparación de 15 particiones (sección 4.2), esa votación da exactamente la misma exactitud que dos etapas + NB-LR solo (0,9023): la ventaja observada en la Mejora 8 venía de una sola partición de validación de 2.400 reseñas.
- **Mejora 9 (robustez):** prueba dos arreglos para la debilidad que encontró el diagnóstico (sección 8): (A) aumento de datos, que agrega al final de la mitad de las reseñas de entrenamiento una oración de una reseña neutral sin cambiar su etiqueta, y (B) usar la última oración *con opinión*, elegida por un detector opinión/hecho (TF-IDF + regresión logística entrenada con oraciones de reseñas neutrales como "hecho" y cláusulas finales de reseñas positivas o negativas como "opinión"). En la regresión logística, A solo empata (−0,10, 7/15), B solo queda justo por debajo del umbral (+0,46, 11/15) y A + B mejora (+0,58, 14/15). En la Mejora 7, A + B mejora +0,59 (14/15). Con una frase neutra al final, los modelos arreglados se mantienen entre 0,895 y 0,907 en prueba, contra 0,49 a 0,70 sin los arreglos. La Mejora 7 + A + B supera a la regresión logística + A + B en 13 de 15 particiones (+0,36), pero esa diferencia depende de la semilla del aumento: en un experimento previo con otra semilla fue de +0,09 (10/15, empate).

## 4. Experimentos complementarios que no mejoraron

Estos experimentos no están en los notebooks del proyecto. Todos se evaluaron con la misma comparación de 15 particiones; las diferencias están en puntos de exactitud frente a la referencia indicada.

### 4.1 Estrategias sobre los modelos de la Mejora 4 (antes de la corrección de la cláusula)

| Experimento | Referencia | Diferencia | Veredicto |
|:---|:---|:---:|:---|
| Bolsa de palabras por tercios del texto | Regresión logística / dos etapas + SVM | −0,03 / +0,07 | empate |
| Marcado de negación (con "nada" que no cancela y corte en "pero") | Regresión logística / dos etapas + SVM | −0,15 / +0,03 | empate |
| NB-SVM + regresión logística con n-gramas de caracteres como bases extra | Apilamiento / dos etapas + SVM | +0,12 / +0,03 | empate (el apilamiento tarda 9 veces más) |
| Entrenar sin las reseñas con cierre ambiguo | Regresión logística / dos etapas + SVM | +0,09 / −0,03 | empate |
| Etapa 2 con Random Forest | Dos etapas + SVM (etapa 2 lineal) | +0,14 | empate |
| Etapa 2 con gradient boosting | Dos etapas + SVM | +0,04 | empate |
| Etapa 2 con K-medias y una regresión por grupo | Dos etapas + SVM | −0,06 | empate |
| Etapa 2 con interacciones de grado 2 | Dos etapas + SVM | −0,35 | empeora |
| Etapa 2 con árbol de decisión | Dos etapas + SVM | −0,57 | empeora |
| Etapa 2 con KNN | Dos etapas + SVM | −1,19 | empeora |

También se buscaron reglas ocultas en las 1.296 reseñas con dos opiniones opuestas. La etiqueta no depende del aspecto comentado, de la opinión sobre el producto nombrado al inicio (46%), de la opinión más intensa (52%) ni de la más larga (46%). La única señal es la posición: la segunda opinión gana el 64% de las veces, y los modelos ya la usan.

### 4.2 Variantes de la Mejora 7 (con la cláusula corregida)

El modelo de la Mejora 7 se copió tal cual de su notebook. Como control, reprodujo exactamente su resultado (0,9023), y la regresión logística corregida el suyo (0,8988).

| Variante | Exactitud 15 particiones | vs. Mejora 7 | Gana en | Veredicto |
|:---|:---:|:---:|:---:|:---|
| Agregar la SVM lineal de la Mejora 4 a la etapa 2 | 0,9026 | +0,04 | 9 de 15 | empate |
| NB-LR con los hiperparámetros de su propia búsqueda (C = 0,1, alpha = 0,5) + SVM, C_etapa2 = 0,03 | 0,9024 | +0,01 | 9 de 15 | empate |
| Votación suave con la regresión logística corregida (idea de la Mejora 8) | 0,9023 | 0,00 | 7 de 15 | empate |
| NB-LR con los hiperparámetros de su propia búsqueda | 0,9021 | −0,02 | 6 de 15 | empate |
| C_etapa2 = 0,03 en lugar de 0,1 | 0,9020 | −0,02 | 5 de 15 | empate |

Todas las combinaciones quedaron entre 0,9019 y 0,9026. Los hiperparámetros no influyen, la SVM ya no aporta una vez que el modelo tiene NB-LR, y la votación no agrega nada a un modelo cuya etapa 2 ya recibe las probabilidades de la regresión logística.

### 4.3 Variantes de la Mejora 9

El modelo de la Mejora 9 se copió de su notebook y, como control, reprodujo exactamente su resultado (0,9081). Se probaron tres ideas:

| Variante | Exactitud 15 particiones | Referencia | Diferencia | Gana en | Veredicto |
|:---|:---:|:---|:---:|:---:|:---|
| Promediar las probabilidades de 3 modelos con distintas semillas del aumento (0, 1 y 2) | 0,9094 | Mejora 9 | +0,13 | 9 de 15 | empate |
| Etapa 2 con los últimos 4 trozos *con opinión* en lugar de los últimos 4 por posición | 0,9075 | Mejora 9 | −0,07 | 5 de 15 | empate |
| Regresión logística + A + B, promedio de 3 semillas | 0,9076 | Regresión logística + A + B | +0,30 | 13 de 15 | mejora |
| Regresión logística + A + B, umbral del detector 0,5 y 25% de reseñas aumentadas | 0,9052 | Regresión logística + A + B | +0,06 | 9 de 15 | empate |
| Regresión logística + A + B, 75% de reseñas aumentadas | 0,9042 | Regresión logística + A + B | −0,03 | 7 de 15 | empate |
| Regresión logística + A + B, umbral del detector 0,3 (con 25%, 50% o 75% aumentadas) | 0,9011 a 0,9016 | Regresión logística + A + B | −0,30 a −0,35 | 3 o 4 de 15 | empeora o empate |
| Regresión logística + A + B, umbral del detector 0,7 (con 25%, 50% o 75% aumentadas) | 0,8760 a 0,8994 | Regresión logística + A + B | −0,51 a −2,86 | 0 a 3 de 15 | empeora |

- **Semillas del aumento:** con una sola semilla, la Mejora 9 da entre 0,9074 y 0,9084 según la semilla, así que su resultado no depende de un sorteo afortunado. Promediar 3 semillas mejora claramente a la regresión logística, que es más sensible al sorteo, pero en el modelo en dos etapas la ganancia (+0,13, unas 3 reseñas de 2.400 por partición) no alcanza el mínimo de 12 de 15 y triplica el tiempo de entrenamiento. Además, la regresión logística con 3 semillas empata con la Mejora 9 (−0,06, 7 de 15).
- **Trozos con opinión:** no aportan. La etapa 2 ya tenía atributos basados en los trozos con opinión, y los bloques de la regresión logística, Naive Bayes y NB-LR ya usan la última oración con opinión.
- **Umbral y proporción:** los valores usados en la Mejora 9 (umbral 0,5, mitad de las reseñas aumentadas) son los mejores o empatan con los mejores. Lo más probable es que con un umbral bajo (0,3) el detector deje pasar más frases sin opinión, y que con uno alto (0,7) salte oraciones que sí la tienen.

La Mejora 9 se mantiene como modelo final: ninguna variante la supera con la regla de decisión, y las que empatan son más complejas o más lentas.

## 5. Análisis de errores y techo

Con el modelo de la Mejora 4, en una predicción fuera de muestra por reseña:

| Grupo | Reseñas | Exactitud | Parte de los errores |
|:---|:---:|:---:|:---:|
| Terminan en una frase de cierre ambigua ("ni fu ni fa", "cada quien que juzgue") | 934 | 0,499 | 39% |
| Conector aditivo ("por otro lado", "de paso", "por cierto", "además") | 1.212 | 0,822 | 18% |
| Conector adversativo ("pero", "aunque", "aun así", "eso sí") | 3.717 | 0,921 | 24% |
| Sin conector de contraste | 2.561 | 0,919 | 17% |
| Neutral | 3.576 | 0,997 | 1% |

- En las reseñas con cierre ambiguo ningún modelo supera el azar: el texto no parece tener información para decidir.
- Los mejores modelos se equivocan en las mismas reseñas: discrepan en solo 1,4% a 3,5% de las predicciones, y entre los dos mejores de la Mejora 8 más del 80% de los errores de uno también son errores del otro. Por eso los ensambles ganan poco.
- Todas las técnicas clásicas probadas convergen en una exactitud de 0,90 a 0,902. Es el techo de la bolsa de palabras para estos datos.

## 6. Lectura del puntaje público

- Todos los puntajes públicos son múltiplos de 1/900, así que el público usa 900 reseñas. Su error estándar es de ±1 punto, y la diferencia entre el mejor envío del grupo (817/900) y el modelo actual (812/900) son 5 reseñas.
- Los dos modelos de la Mejora 9 sacaron 826/900 y 825/900. Según la comparación de 15 particiones se esperaban unos 817 y 814 aciertos, así que los dos quedaron cerca de 1 error estándar por encima. No son dos confirmaciones independientes: sus envíos coinciden en 2.898 de las 3.000 predicciones, así que comparten casi toda la suerte del público. La estimación más honesta para el puntaje final sigue siendo la de 15 particiones (alrededor de 0,908).
- Si se elige el mejor de varios envíos por su puntaje público, ese puntaje queda inflado por la selección. Un puntaje de 0,921 (829/900) es muy improbable con un modelo de exactitud real de 0,90 en un solo envío (2%), pero se vuelve posible si se elige entre muchos.

## 7. Modelo elegido

**Dos etapas + NB-LR con los arreglos de robustez (Mejora 9, `dos_etapas_nblr_robusto`)** es el mejor modelo en la comparación justa: 0,9081 en 15 particiones, +0,59 puntos sobre la Mejora 7, ganando en 14 de 15. Además es robusto a frases sin opinión al final de la reseña, que era la principal limitación del modelo anterior. También tiene el mejor puntaje público del grupo: 0,91777 (826/900), 9 reseñas más que la Mejora 7. Como cumple los dos criterios, es el modelo que se sube a Bloque Neón (`models/dos_etapas_nblr_robusto.joblib`). La regresión logística + A + B quedó a 1 reseña (825/900), una diferencia que es ruido.

Antes de la Mejora 9, **dos etapas + NB-LR (Mejora 7)** era el mejor modelo según los dos criterios disponibles:

- La mayor exactitud en la comparación justa de 15 particiones: 0,9023, +0,56 puntos sobre el modelo actual, ganando en 14 de 15.
- El mejor puntaje público del grupo en Kaggle: 0,90777 (817/900).

Su ventaja sobre las otras alternativas fuertes (Mejora 4, NB-LR, votación suave de la Mejora 8) es de décimas y está dentro del ruido. Además, el modelo está en su punto óptimo: ninguna variante probada (sección 4.2) se aleja más de 0,04 puntos de su exactitud.

## 8. Diagnóstico del modelo final (`Diagnostico_Modelo_Final.ipynb`)

La primera versión de este diagnóstico se hizo con el modelo de la Mejora 7 y encontró su principal debilidad: una frase neutra al final bajaba la exactitud a entre 0,63 y 0,71, porque los modelos aprendieron asociaciones espurias entre las palabras de la última oración y la etiqueta (por ejemplo, las reseñas cuya última oración habla de uso diario son 268 positivas contra 163 negativas en entrenamiento). Eso motivó la Mejora 9. La versión actual revisa el modelo final (Mejora 9) con 12.000 predicciones fuera de muestra (validación cruzada de 5 particiones):

- **Sobreajuste:** 98,4% de exactitud en entrenamiento contra 91,0% fuera de muestra (7,4 puntos; la Mejora 7 tenía 8,8). La diferencia se concentra en las reseñas ambiguas, que el modelo memoriza (93% contra 54% en las de cierre ambiguo); en las neutrales es prácticamente cero. La validación es estable (±0,5 puntos).
- **Fugas de información (prueba de permutación):** con las etiquetas barajadas, el modelo final queda en 0,352 y la regresión logística robusta en 0,328, al nivel del azar (clase más frecuente: 0,351). El aumento de datos y el detector no filtran información de validación. La regresión logística memoriza el 85% de las etiquetas al azar en entrenamiento, lo que muestra que su exactitud de entrenamiento no indica su desempeño real.
- **Kaggle:** el puntaje público (0,9178) está 1,0 errores estándar por encima de la validación de 15 particiones (0,9081), y el modelo se eligió sin mirarlo. Para el puntaje final se espera entre 0,896 y 0,921.
- **Curva de aprendizaje:** la validación sube de 0,890 (1.920 reseñas) a 0,910 (9.600). Más datos del mismo tipo ayudarían solo décimas.
- **Sesgo entre clases:** no hay. Predice 35,2% negativo, 29,8% neutral y 34,9% positivo (real: 35,1, 29,8 y 35,1), con errores simétricos (524 contra 543) y solo 10 errores relacionados con `neutral`.
- **Calibración:** con más de 0,9 de confianza (81% de las reseñas) acierta el 98%. En el 19% restante es demasiado optimista y concentra el 84% de los errores.
- **Subgrupos:** no hay sesgo por tildes (0,911 contra 0,908) ni por longitud. Rinde peor en reseñas de 5 o más oraciones (0,797), porque en ellas se concentran los cierres ambiguos (25,7%, contra 1,6% en las de 1 a 3 oraciones) y los conectores aditivos.
- **Robustez:** con una frase neutra al final la exactitud baja solo entre 0,1 y 0,2 puntos (entre 0,908 y 0,909 con cuatro frases distintas). Quitar tildes o la primera oración no lo afecta, y los errores de digitación le siguen restando cerca de 1 punto.
- **Datos de la competencia:** se parecen a los de entrenamiento (AUC de 0,529 para distinguirlos), así que el puntaje final debería ser cercano a la validación cruzada.

## 9. Limitaciones y trabajo futuro

- La representación por bloques y el modelo por oración dependen de la estructura de estas reseñas (la opinión al final, conectores fijos). El diagnóstico lo confirma: una frase neutra al final basta para bajar la exactitud a menos de 0,71. La Mejora 9 lo corrige para frases agregadas al final, pero no se midió la robustez ante otros cambios, como frases neutras en medio del texto. Además, su detector de opinión solo reconoce opiniones explícitas y puede tratar una queja implícita como un hecho.
- La bolsa de palabras no entiende el significado: "cumple de sobra" y "valió la pena" son atributos distintos. Los embeddings y los modelos de la segunda parte del curso apuntan a esa limitación. Aun así, los errores en reseñas con cierre ambiguo (cerca del 4% de todas las reseñas) seguirían siendo un límite de los datos.
