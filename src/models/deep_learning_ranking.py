import numpy as np
import pandas as pd


def recomendar_deep_learning_ranking(
    usuario_id,
    historial,
    movies,
    modelo,
    usuario_a_indice,
    pelicula_a_indice,
    n=10
):
    """
    Genera recomendaciones Top-N para un usuario existente
    utilizando el modelo Deep Learning orientado a ranking.

    La salida del modelo representa una puntuación de
    relevancia entre 0 y 1.
    """

    # -----------------------------------------------------
    # COMPROBAR QUE EL USUARIO EXISTE EN EL MAPPING
    # -----------------------------------------------------

    if usuario_id not in usuario_a_indice:
        return pd.DataFrame()


    usuario_idx = usuario_a_indice[
        usuario_id
    ]


    # -----------------------------------------------------
    # PELÍCULAS YA VALORADAS
    # -----------------------------------------------------

    peliculas_vistas = set(
        historial["movieId"]
    )


    # -----------------------------------------------------
    # PELÍCULAS CANDIDATAS
    #
    # Solo podemos utilizar películas que existan en el
    # mapping utilizado durante el entrenamiento.
    # -----------------------------------------------------

    peliculas_candidatas = [
        movie_id
        for movie_id in pelicula_a_indice.keys()
        if movie_id not in peliculas_vistas
    ]


    if len(peliculas_candidatas) == 0:
        return pd.DataFrame()


    # -----------------------------------------------------
    # CONVERTIR PELÍCULAS A ÍNDICES INTERNOS
    # -----------------------------------------------------

    peliculas_idx = np.array(
        [
            pelicula_a_indice[movie_id]
            for movie_id in peliculas_candidatas
        ],
        dtype="int32"
    )


    # -----------------------------------------------------
    # REPETIR ÍNDICE DEL USUARIO
    # -----------------------------------------------------

    usuarios_idx = np.full(
        shape=len(peliculas_idx),
        fill_value=usuario_idx,
        dtype="int32"
    )


    # -----------------------------------------------------
    # PREDICCIONES DE RELEVANCIA
    # -----------------------------------------------------

    puntuaciones = modelo.predict(
        [
            usuarios_idx,
            peliculas_idx
        ],
        batch_size=1024,
        verbose=0
    ).flatten()


    # -----------------------------------------------------
    # CREAR DATAFRAME DE RESULTADOS
    # -----------------------------------------------------

    recomendaciones = pd.DataFrame(
        {
            "movieId": peliculas_candidatas,
            "relevancia_dl": puntuaciones
        }
    )


    # -----------------------------------------------------
    # AÑADIR INFORMACIÓN DE LAS PELÍCULAS
    # -----------------------------------------------------

    recomendaciones = recomendaciones.merge(
        movies[
            [
                "movieId",
                "title",
                "genres"
            ]
        ],
        on="movieId",
        how="left"
    )


    # -----------------------------------------------------
    # ORDENAR POR RELEVANCIA
    # -----------------------------------------------------

    recomendaciones = (
        recomendaciones
        .sort_values(
            by="relevancia_dl",
            ascending=False
        )
        .head(n)
        .reset_index(drop=True)
    )


    return recomendaciones[
        [
            "movieId",
            "title",
            "genres",
            "relevancia_dl"
        ]
    ]