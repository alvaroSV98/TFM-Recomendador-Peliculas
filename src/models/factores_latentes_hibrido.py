import pandas as pd


def recomendar_hibrido(
    usuario_id,
    historial,
    ratings,
    movies,
    P,
    Q,
    bias_usuarios,
    bias_peliculas,
    media_global,
    user_to_idx,
    movie_to_idx,
    n=10,
    alpha=0.20
):

    # -----------------------------------------------------
    # CALCULAR POPULARIDAD DE LAS PELÍCULAS
    # -----------------------------------------------------

    popularidad = (
        ratings
        .groupby("movieId")
        .size()
    )

    popularidad_normalizada = (
        (popularidad - popularidad.min())
        /
        (popularidad.max() - popularidad.min())
    )


    # -----------------------------------------------------
    # PELÍCULAS YA VISTAS
    # -----------------------------------------------------

    peliculas_vistas = set(
        historial["movieId"]
    )

    resultados = []

    u = user_to_idx[usuario_id]


    # -----------------------------------------------------
    # CALCULAR SCORE HÍBRIDO
    # -----------------------------------------------------

    for pelicula_id in movie_to_idx:

        if pelicula_id in peliculas_vistas:
            continue

        i = movie_to_idx[pelicula_id]

        # Score de factores latentes
        score_mf = (
            media_global
            + bias_usuarios[u]
            + bias_peliculas[i]
            + P[u] @ Q[i]
        )

        # Limitar la predicción a la escala original de MovieLens
        score_mf = min(max(score_mf, 0.5), 5.0)

        # Normalizar el score de factores latentes a [0, 1]
        score_mf_norm = (
            (score_mf - 0.5)
            /
            (5.0 - 0.5)
        )

        # Score de popularidad ya normalizado a [0, 1]
        score_pop = popularidad_normalizada.get(
            pelicula_id,
            0.0
        )

        # Combinación híbrida sobre una escala común
        score_hibrido = (
            alpha * score_mf_norm
            +
            (1 - alpha) * score_pop
        )

        resultados.append(
            (
                pelicula_id,
                score_hibrido
            )
        )


    # -----------------------------------------------------
    # TOP-N
    # -----------------------------------------------------

    resultados = sorted(
        resultados,
        key=lambda x: x[1],
        reverse=True
    )[:n]

    recomendaciones = pd.DataFrame(
        resultados,
        columns=[
            "movieId",
            "score_hibrido"
        ]
    )

    recomendaciones = recomendaciones.merge(
        movies,
        on="movieId"
    )

    return recomendaciones