import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


def preparar_filtrado_colaborativo(ratings):
    """
    Prepara las estructuras necesarias para el
    filtrado colaborativo basado en usuarios.
    """

    # -----------------------------------------------------
    # MATRIZ USUARIO-PELÍCULA
    # -----------------------------------------------------

    matriz_usuario_pelicula = ratings.pivot(
        index="userId",
        columns="movieId",
        values="rating"
    )


    # -----------------------------------------------------
    # MEDIA DE CADA USUARIO
    # -----------------------------------------------------

    media_por_usuario = (
        matriz_usuario_pelicula
        .mean(axis=1)
    )


    # -----------------------------------------------------
    # NORMALIZACIÓN
    # -----------------------------------------------------

    matriz_normalizada = (
        matriz_usuario_pelicula
        .sub(
            media_por_usuario,
            axis=0
        )
    )


    # -----------------------------------------------------
    # MATRIZ PARA CALCULAR SIMILITUD
    # -----------------------------------------------------

    matriz_similitud = (
        matriz_normalizada
        .fillna(0)
    )


    # -----------------------------------------------------
    # SIMILITUD COSENO
    # -----------------------------------------------------

    similitud_usuarios = cosine_similarity(
        matriz_similitud
    )


    similitud_usuarios = pd.DataFrame(
        similitud_usuarios,
        index=matriz_usuario_pelicula.index,
        columns=matriz_usuario_pelicula.index
    )


    return (
        matriz_usuario_pelicula,
        matriz_normalizada,
        media_por_usuario,
        similitud_usuarios
    )



def predecir_filtrado_colaborativo(
    usuario_id,
    pelicula_id,
    matriz_normalizada,
    media_por_usuario,
    similitud_usuarios,
    k=10,
    min_vecinos=3
):
    """
    Predice la valoración de una película para un usuario
    mediante filtrado colaborativo basado en usuarios.
    """

    # -----------------------------------------------------
    # COMPROBAR USUARIO
    # -----------------------------------------------------

    if usuario_id not in similitud_usuarios.index:
        return np.nan


    # -----------------------------------------------------
    # COMPROBAR PELÍCULA
    # -----------------------------------------------------

    if pelicula_id not in matriz_normalizada.columns:
        return np.nan


    # -----------------------------------------------------
    # SIMILITUDES DEL USUARIO
    # -----------------------------------------------------

    similitudes_usuario = (
        similitud_usuarios
        .loc[usuario_id]
        .drop(usuario_id)
    )


    # -----------------------------------------------------
    # SELECCIONAR K VECINOS
    # -----------------------------------------------------

    vecinos = (
        similitudes_usuario
        .sort_values(
            ascending=False
        )
        .head(k)
    )


    # -----------------------------------------------------
    # DESVIACIONES DE LOS VECINOS
    # -----------------------------------------------------

    desviaciones = (
        matriz_normalizada
        .loc[
            vecinos.index,
            pelicula_id
        ]
        .dropna()
    )


    # -----------------------------------------------------
    # EVIDENCIA MÍNIMA
    # -----------------------------------------------------

    if len(desviaciones) < min_vecinos:
        return np.nan


    # -----------------------------------------------------
    # SIMILITUDES VÁLIDAS
    # -----------------------------------------------------

    similitudes_validas = (
        vecinos.loc[
            desviaciones.index
        ]
    )


    suma_similitudes = (
        similitudes_validas
        .abs()
        .sum()
    )


    if suma_similitudes == 0:
        return np.nan


    # -----------------------------------------------------
    # DESVIACIÓN PREDICHA
    # -----------------------------------------------------

    desviacion_predicha = (
        (
            desviaciones
            *
            similitudes_validas
        ).sum()
        /
        suma_similitudes
    )


    # -----------------------------------------------------
    # VOLVER A LA ESCALA ORIGINAL
    # -----------------------------------------------------

    prediccion = (
        media_por_usuario.loc[usuario_id]
        +
        desviacion_predicha
    )


    # -----------------------------------------------------
    # LIMITAR A ESCALA MOVIELENS
    # -----------------------------------------------------

    prediccion = np.clip(
        prediccion,
        0.5,
        5.0
    )


    return float(prediccion)



def recomendar_filtrado_colaborativo(
    usuario_id,
    historial,
    movies,
    matriz_normalizada,
    media_por_usuario,
    similitud_usuarios,
    k=10,
    min_vecinos=3,
    n=10
):
    """
    Genera el Top-N de recomendaciones para un usuario
    mediante filtrado colaborativo basado en usuarios.
    """

    # -----------------------------------------------------
    # PELÍCULAS YA VISTAS
    # -----------------------------------------------------

    peliculas_vistas = set(
        historial["movieId"]
    )


    # -----------------------------------------------------
    # PELÍCULAS CANDIDATAS
    # -----------------------------------------------------

    peliculas_candidatas = (
        matriz_normalizada
        .columns
        .difference(
            peliculas_vistas
        )
    )


    # -----------------------------------------------------
    # CALCULAR PREDICCIONES
    # -----------------------------------------------------

    resultados = []


    for pelicula_id in peliculas_candidatas:

        prediccion = predecir_filtrado_colaborativo(
            usuario_id=usuario_id,
            pelicula_id=pelicula_id,
            matriz_normalizada=matriz_normalizada,
            media_por_usuario=media_por_usuario,
            similitud_usuarios=similitud_usuarios,
            k=k,
            min_vecinos=min_vecinos
        )


        if not np.isnan(prediccion):

            resultados.append(
                {
                    "movieId": pelicula_id,
                    "prediccion_cf": prediccion
                }
            )


    # -----------------------------------------------------
    # COMPROBAR RESULTADOS
    # -----------------------------------------------------

    if len(resultados) == 0:
        return pd.DataFrame()


    recomendaciones = pd.DataFrame(
        resultados
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
    # TOP-N
    # -----------------------------------------------------

    recomendaciones = (
        recomendaciones
        .sort_values(
            by="prediccion_cf",
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
            "prediccion_cf"
        ]
    ]