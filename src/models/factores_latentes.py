import pandas as pd


def recomendar_factores_latentes(
    usuario_id,
    historial,
    movies,
    P,
    Q,
    bias_usuarios,
    bias_peliculas,
    media_global,
    user_to_idx,
    movie_to_idx,
    n=10
):

    peliculas_vistas = set(
        historial["movieId"]
    )

    predicciones = []

    u = user_to_idx[usuario_id]

    for pelicula_id in movie_to_idx:

        if pelicula_id in peliculas_vistas:
            continue

        i = movie_to_idx[pelicula_id]

        prediccion = (
            media_global
            + bias_usuarios[u]
            + bias_peliculas[i]
            + P[u] @ Q[i]
        )

        predicciones.append(
            (
                pelicula_id,
                prediccion
            )
        )

    predicciones = sorted(
        predicciones,
        key=lambda x: x[1],
        reverse=True
    )[:n]

    recomendaciones = pd.DataFrame(
        predicciones,
        columns=[
            "movieId",
            "prediccion"
        ]
    )

    recomendaciones = recomendaciones.merge(
        movies,
        on="movieId"
    )

    return recomendaciones