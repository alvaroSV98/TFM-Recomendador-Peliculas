# Sistema recomendador de películas

Proyecto desarrollado como Trabajo de Fin de Máster en Inteligencia Artificial.

El objetivo del proyecto es desarrollar y comparar diferentes técnicas de recomendación de películas utilizando el dataset MovieLens Latest Small. Para ello se han implementado varios enfoques de recomendación, desde métodos clásicos de filtrado colaborativo hasta modelos de Deep Learning.

El proyecto incluye tanto el desarrollo y evaluación de los modelos como una aplicación web desarrollada con Streamlit para generar recomendaciones de películas.

## Objetivo del proyecto

El objetivo principal es analizar, desarrollar y comparar diferentes enfoques de sistemas recomendadores aplicados al dominio cinematográfico.

Para ello se han trabajado varios modelos con características distintas:

- Filtrado colaborativo basado en usuarios mediante similitud del coseno.
- Modelo de factores latentes mediante factorización matricial.
- Modelo de Deep Learning orientado a predicción de valoraciones.
- Modelo de Deep Learning orientado a ranking y relevancia.
- Estrategias híbridas y de cold start para complementar el sistema.

Además, se ha desarrollado una aplicación web con Streamlit que permite utilizar los distintos modelos de recomendación de forma interactiva.

## Dataset

Para el desarrollo del proyecto se ha utilizado el dataset MovieLens Latest Small, proporcionado por GroupLens.

El conjunto de datos contiene:

- 100.836 valoraciones.
- 610 usuarios.
- 9.742 películas.
- Valoraciones comprendidas entre 0,5 y 5 estrellas.

La información principal utilizada en los modelos procede de los archivos `ratings.csv` y `movies.csv`, mientras que el resto de archivos del dataset se conserva dentro del repositorio como parte de la distribución original de MovieLens.

El dataset se encuentra en:

`data/raw/ml-latest-small/`

MovieLens Latest Small ha sido desarrollado por GroupLens Research.

Referencia:

F. Maxwell Harper y Joseph A. Konstan. *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems, 2015.

## Modelos desarrollados

### Filtrado colaborativo basado en usuarios

Se ha implementado un modelo de filtrado colaborativo basado en usuarios utilizando similitud del coseno.

El modelo trabaja con valoraciones centradas respecto a la media de cada usuario y utiliza los vecinos más similares para estimar la valoración de películas no vistas.

### Factores latentes

Se ha desarrollado un modelo de factorización matricial basado en factores latentes.

Este enfoque representa a usuarios y películas mediante vectores de características latentes y aprende sus relaciones a partir de las valoraciones disponibles.

También se evaluó una extensión híbrida que combina la puntuación del modelo con información de popularidad.

### Deep Learning para predicción de valoraciones

Se ha desarrollado una red neuronal que combina embeddings de usuarios y películas junto con información adicional relacionada con los géneros.

Su objetivo es estimar la valoración que un usuario podría asignar a una película.

### Deep Learning para ranking

Además del enfoque de predicción de valoraciones, se desarrolló un segundo modelo de Deep Learning orientado directamente a ranking.

Este modelo estima la relevancia de cada película para un usuario y permite ordenar las películas candidatas según dicha puntuación.

### Cold start

Para usuarios nuevos, que todavía no disponen de historial dentro de MovieLens, se ha implementado una estrategia específica de cold start.

El usuario puede valorar inicialmente varias películas y, a partir de estas preferencias, el sistema combina información colaborativa, contenido y popularidad para generar recomendaciones.

## Evaluación

Los modelos se han evaluado utilizando métricas adaptadas a los distintos objetivos del proyecto.

Para los modelos orientados a predicción de valoraciones se utilizaron principalmente:

- MAE.
- RMSE.

En la comparación realizada sobre un subconjunto común de predicciones se obtuvieron los siguientes resultados:

| Modelo | MAE | RMSE |
|---|---:|---:|
| Filtrado colaborativo | 0,6325 | 0,8257 |
| Factores latentes | 0,6003 | 0,7791 |
| Deep Learning | 0,5874 | 0,7681 |

En los modelos orientados a ranking se utilizaron métricas como Hit Rate@10 y NDCG@10.

El modelo final de Deep Learning orientado a ranking obtuvo:

- HR@10: 0,5612
- NDCG@10: 0,4388

Los distintos resultados no deben interpretarse siempre como una comparación directa entre todos los modelos, ya que algunos experimentos utilizan protocolos de evaluación y conjuntos de candidatos diferentes.

## Aplicación web

El proyecto incluye una aplicación desarrollada con Streamlit que permite utilizar los modelos de recomendación de forma interactiva.

La aplicación contempla dos escenarios principales:

- Usuarios existentes de MovieLens, para los que se pueden utilizar los distintos modelos entrenados.
- Usuarios nuevos, para los que se genera una recomendación inicial a partir de las películas valoradas durante la sesión.

Los modelos disponibles para usuarios existentes incluyen:

- Filtrado colaborativo.
- Factores latentes.
- Factores latentes combinados con popularidad.
- Deep Learning.
- Deep Learning orientado a ranking.

## Estructura del proyecto

```text
TFM-Recomendador-Peliculas/
│
├── app/
│   └── app.py
│
├── data/
│   └── raw/
│       └── ml-latest-small/
│
├── models/
│   ├── factores_latentes.pkl
│   ├── modelo_deep_learning.keras
│   ├── modelo_deep_learning_ranking.keras
│   ├── usuario_a_indice.pkl
│   ├── pelicula_a_indice.pkl
│   ├── columnas_generos.pkl
│   ├── preferencias_usuario.csv
│   ├── usuario_a_indice_ranking.pkl
│   └── pelicula_a_indice_ranking.pkl
│
├── notebooks/
│   ├── 01_exploracion_movielens.ipynb
│   ├── 02_filtrado_colaborativo.ipynb
│   ├── 03_evaluacion_filtrado_colaborativo.ipynb
│   ├── 04_factores_latentes.ipynb
│   ├── 05_evaluacion_factores_latentes.ipynb
│   ├── 06_factores_latentes_hibrido.ipynb
│   ├── 07_evaluacion_factores_latentes_hibrido.ipynb
│   ├── 08_deep_learning.ipynb
│   ├── 09_evaluacion_deep_learning.ipynb
│   ├── 10_deep_learning_ranking.ipynb
│   └── 11_evaluacion_comparativa_final.ipynb
│
├── src/
│   ├── __init__.py
│   └── models/
│       ├── __init__.py
│       ├── cold_start.py
│       ├── deep_learning.py
│       ├── deep_learning_ranking.py
│       ├── factores_latentes.py
│       ├── factores_latentes_hibrido.py
│       └── filtrado_colaborativo.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Instalación

Para instalar las dependencias del proyecto:

```bash
pip install -r requirements.txt
```

## Ejecución de la aplicación

Desde la carpeta raíz del proyecto:

```bash
streamlit run app/app.py
```

## Tecnologías utilizadas

- Python
- Pandas
- NumPy
- scikit-learn
- TensorFlow / Keras
- Matplotlib
- Streamlit
- Jupyter Notebook

## Autor

Álvaro Sierras Valdés
Proyecto realizado como Trabajo de Fin de Máster en Inteligencia Artificial.
