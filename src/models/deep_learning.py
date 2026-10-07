import numpy as np
import pandas as pd


def preparar_generos_movies(
    movies,
    columnas_generos
):

    movies_generos = movies.copy()

    for genero in columnas_generos:

        movies_generos[genero] = (
            movies_generos["genres"]
            .str.split("|")
            .apply(
                lambda lista_generos:
                1.0 if genero in lista_generos else 0.0
            )
        )

    return movies_generos


def recomendar_deep_learning(
    usuario_id,
    historial,
    movies,
    modelo,
    usuario_a_indice,
    pelicula_a_indice,
    columnas_generos,
    preferencias_usuario,
    n=10
):

    # -----------------------------------------------------
    # COMPROBAR USUARIO
    # -----------------------------------------------------

    if usuario_id not in usuario_a_indice:
        return pd.DataFrame()


    # -----------------------------------------------------
    # PREPARAR GÉNEROS DE LAS PELÍCULAS
    # -----------------------------------------------------

    movies_generos = preparar_generos_movies(
        movies,
        columnas_generos
    )


    # -----------------------------------------------------
    # PELÍCULAS YA VISTAS POR EL USUARIO
    # -----------------------------------------------------

    peliculas_vistas = set(
        historial["movieId"]
    )


    # -----------------------------------------------------
    # SELECCIONAR PELÍCULAS CANDIDATAS
    # -----------------------------------------------------

    candidatas = movies_generos[
        (~movies_generos["movieId"].isin(peliculas_vistas))
        &
        (movies_generos["movieId"].isin(pelicula_a_indice))
    ].copy()

    if candidatas.empty:
        return pd.DataFrame()


    # -----------------------------------------------------
    # ÍNDICE INTERNO DEL USUARIO
    # -----------------------------------------------------

    usuario_idx = usuario_a_indice[
        usuario_id
    ]


    # -----------------------------------------------------
    # PREFERENCIAS DEL USUARIO
    # -----------------------------------------------------

    preferencias_fila = preferencias_usuario[
        preferencias_usuario["userId"] == usuario_id
    ]

    if preferencias_fila.empty:
        return pd.DataFrame()


    columnas_preferencias = [
        f"pref_{genero}"
        for genero in columnas_generos
    ]


    vector_preferencias = preferencias_fila[
        columnas_preferencias
    ].values.astype(
        "float32"
    )


    # -----------------------------------------------------
    # NÚMERO DE PELÍCULAS CANDIDATAS
    # -----------------------------------------------------

    num_candidatas = len(
        candidatas
    )


    # -----------------------------------------------------
    # PREPARAR ENTRADA DE USUARIOS
    # -----------------------------------------------------

    usuarios_input = np.full(
        (
            num_candidatas,
            1
        ),
        usuario_idx,
        dtype="float32"
    )


    # -----------------------------------------------------
    # PREPARAR ENTRADA DE PELÍCULAS
    # -----------------------------------------------------

    peliculas_input = np.array(
        [
            pelicula_a_indice[movie_id]
            for movie_id in candidatas["movieId"]
        ],
        dtype="float32"
    ).reshape(
        -1,
        1
    )


    # -----------------------------------------------------
    # PREPARAR ENTRADA DE GÉNEROS
    # -----------------------------------------------------

    generos_input = candidatas[
        columnas_generos
    ].values.astype(
        "float32"
    )


    # -----------------------------------------------------
    # PREPARAR ENTRADA DE PREFERENCIAS
    # -----------------------------------------------------

    preferencias_input = np.repeat(
        vector_preferencias,
        num_candidatas,
        axis=0
    )


    # -----------------------------------------------------
    # PREDICCIÓN EN BLOQUE
    # -----------------------------------------------------

    predicciones = modelo.predict(
        {
            "usuario": usuarios_input,
            "pelicula": peliculas_input,
            "generos": generos_input,
            "preferencias": preferencias_input
        },
        verbose=0
    ).reshape(
        -1
    )


    # -----------------------------------------------------
    # AÑADIR PREDICCIONES A LAS PELÍCULAS
    # -----------------------------------------------------

    candidatas[
        "prediccion_dl"
    ] = predicciones


    # -----------------------------------------------------
    # TOP-N RECOMENDACIONES
    # -----------------------------------------------------

    recomendaciones = (
        candidatas
        .sort_values(
            by="prediccion_dl",
            ascending=False
        )
        .head(n)
        [
            [
                "movieId",
                "title",
                "genres",
                "prediccion_dl"
            ]
        ]
        .reset_index(
            drop=True
        )
    )


    return recomendaciones