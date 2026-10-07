import numpy as np
import pandas as pd


def buscar_vecinos_nuevo_usuario(
    valoraciones,
    ratings,
    min_comunes=3,
    constante_confianza=5
):
    resultados = []

    peliculas_nuevo = set(
        valoraciones.keys()
    )

    for usuario_id, datos_usuario in ratings.groupby("userId"):

        valoraciones_usuario = (
            datos_usuario[
                datos_usuario["movieId"].isin(peliculas_nuevo)
            ]
            .set_index("movieId")["rating"]
        )

        peliculas_comunes = list(
            set(valoraciones_usuario.index)
            &
            peliculas_nuevo
        )

        numero_comunes = len(
            peliculas_comunes
        )

        if numero_comunes < min_comunes:
            continue


        nuevo = np.array(
            [
                valoraciones[movie_id]
                for movie_id in peliculas_comunes
            ],
            dtype=float
        )

        existente = np.array(
            [
                valoraciones_usuario[movie_id]
                for movie_id in peliculas_comunes
            ],
            dtype=float
        )


        nuevo_centrado = (
            nuevo - nuevo.mean()
        )

        existente_centrado = (
            existente - existente.mean()
        )

        denominador = (
            np.linalg.norm(nuevo_centrado)
            *
            np.linalg.norm(existente_centrado)
        )

        if denominador == 0:
            correlacion = 0.0
        else:
            correlacion = (
                np.dot(
                    nuevo_centrado,
                    existente_centrado
                )
                /
                denominador
            )


        mae = np.mean(
            np.abs(
                nuevo - existente
            )
        )

        similitud_notas = (
            1.0 / (1.0 + mae)
        )


        similitud_base = (
            0.7 * correlacion
            +
            0.3 * similitud_notas
        )


        factor_confianza = (
            numero_comunes
            /
            (
                numero_comunes
                +
                constante_confianza
            )
        )


        similitud_ajustada = (
            similitud_base
            *
            factor_confianza
        )


        resultados.append(
            {
                "userId": usuario_id,
                "correlacion": correlacion,
                "mae": mae,
                "similitud_notas": similitud_notas,
                "peliculas_comunes": numero_comunes,
                "factor_confianza": factor_confianza,
                "similitud_ajustada": similitud_ajustada
            }
        )


    vecinos = pd.DataFrame(
        resultados
    )

    if vecinos.empty:
        return vecinos


    vecinos = (
        vecinos
        .sort_values(
            by=[
                "similitud_ajustada",
                "peliculas_comunes"
            ],
            ascending=[
                False,
                False
            ]
        )
        .reset_index(drop=True)
    )

    return vecinos



def construir_perfil_contenido(
    valoraciones,
    movies,
    valor_neutral=3.0
):
    numero_peliculas = len(
        valoraciones
    )

    if numero_peliculas == 0:
        return {}


    datos = pd.DataFrame(
        [
            {
                "movieId": movie_id,
                "rating": rating
            }
            for movie_id, rating in valoraciones.items()
        ]
    )


    datos = datos.merge(
        movies[
            [
                "movieId",
                "genres"
            ]
        ],
        on="movieId",
        how="left"
    )


    perfil = {}


    for _, fila in datos.iterrows():

        preferencia = (
            float(fila["rating"])
            -
            valor_neutral
        )

        generos = (
            fila["genres"]
            .split("|")
        )


        for genero in generos:

            if genero == "(no genres listed)":
                continue

            if genero not in perfil:
                perfil[genero] = 0.0

            perfil[genero] += preferencia


    for genero in perfil:

        perfil[genero] = (
            perfil[genero]
            /
            numero_peliculas
        )


    return perfil



def calcular_score_contenido(
    generos_pelicula,
    perfil_contenido,
    valor_neutral=3.0
):
    generos = [
        genero
        for genero in generos_pelicula.split("|")
        if genero != "(no genres listed)"
    ]

    if len(generos) == 0:
        return valor_neutral


    afinidad = sum(
        perfil_contenido.get(
            genero,
            0.0
        )
        for genero in generos
    )


    afinidad_media = (
        afinidad
        /
        len(generos)
    )


    score = (
        valor_neutral
        +
        afinidad_media
    )


    return float(
        np.clip(
            score,
            0.5,
            5.0
        )
    )



def recomendar_cold_start(
    valoraciones,
    ratings,
    movies,
    media_global,
    min_comunes=3,
    n_vecinos=10,
    n_recomendaciones=10,
    min_vecinos=3,
    peso_colaborativo=0.55,
    peso_contenido=0.30,
    peso_global=0.15,
    m=20,
    constante_confianza_vecinos=3
):
    suma_pesos = (
        peso_colaborativo
        +
        peso_contenido
        +
        peso_global
    )

    if not np.isclose(
        suma_pesos,
        1.0
    ):
        raise ValueError(
            "Los pesos deben sumar 1."
        )


    # -----------------------------------------------------
    # BUSCAR VECINOS
    # -----------------------------------------------------

    vecinos = buscar_vecinos_nuevo_usuario(
        valoraciones=valoraciones,
        ratings=ratings,
        min_comunes=min_comunes,
        constante_confianza=5
    )


    if vecinos.empty:
        return pd.DataFrame()


    mejores_vecinos = (
        vecinos
        .head(n_vecinos)
        .copy()
    )


    ids_vecinos = set(
        mejores_vecinos["userId"]
    )


    # -----------------------------------------------------
    # PERFIL DE CONTENIDO
    # -----------------------------------------------------

    perfil_contenido = construir_perfil_contenido(
        valoraciones=valoraciones,
        movies=movies
    )


    # -----------------------------------------------------
    # RATINGS DE LOS VECINOS
    # -----------------------------------------------------

    ratings_vecinos = ratings[
        ratings["userId"].isin(
            ids_vecinos
        )
    ].copy()


    peliculas_vistas = set(
        valoraciones.keys()
    )


    ratings_vecinos = ratings_vecinos[
        ~ratings_vecinos["movieId"].isin(
            peliculas_vistas
        )
    ].copy()


    ratings_vecinos = ratings_vecinos.merge(
        mejores_vecinos[
            [
                "userId",
                "similitud_ajustada"
            ]
        ],
        on="userId",
        how="left"
    )


    ratings_vecinos = ratings_vecinos[
        ratings_vecinos["similitud_ajustada"] > 0
    ].copy()


    if ratings_vecinos.empty:
        return pd.DataFrame()


    # -----------------------------------------------------
    # PARTE COLABORATIVA
    # -----------------------------------------------------

    ratings_vecinos["rating_ponderado"] = (
        ratings_vecinos["rating"]
        *
        ratings_vecinos["similitud_ajustada"]
    )


    recomendaciones = (
        ratings_vecinos
        .groupby("movieId")
        .agg(
            suma_ponderada=(
                "rating_ponderado",
                "sum"
            ),
            suma_similitudes=(
                "similitud_ajustada",
                "sum"
            ),
            numero_vecinos=(
                "userId",
                "nunique"
            )
        )
        .reset_index()
    )


    recomendaciones["prediccion_vecinos"] = (
        recomendaciones["suma_ponderada"]
        /
        recomendaciones["suma_similitudes"]
    )


    recomendaciones = recomendaciones[
        recomendaciones["numero_vecinos"] >= min_vecinos
    ].copy()


    if recomendaciones.empty:
        return pd.DataFrame()


    recomendaciones["confianza_vecinos"] = (
        recomendaciones["numero_vecinos"]
        /
        (
            recomendaciones["numero_vecinos"]
            +
            constante_confianza_vecinos
        )
    )


    recomendaciones["score_colaborativo"] = (
        recomendaciones["confianza_vecinos"]
        *
        recomendaciones["prediccion_vecinos"]
        +
        (
            1
            -
            recomendaciones["confianza_vecinos"]
        )
        *
        media_global
    )


    # -----------------------------------------------------
    # PARTE GLOBAL
    # -----------------------------------------------------

    estadisticas_peliculas = (
        ratings
        .groupby("movieId")
        .agg(
            media_pelicula=(
                "rating",
                "mean"
            ),
            numero_valoraciones=(
                "rating",
                "count"
            )
        )
        .reset_index()
    )


    estadisticas_peliculas["rating_regularizado"] = (
        (
            estadisticas_peliculas["numero_valoraciones"]
            /
            (
                estadisticas_peliculas["numero_valoraciones"]
                +
                m
            )
        )
        *
        estadisticas_peliculas["media_pelicula"]
        +
        (
            m
            /
            (
                estadisticas_peliculas["numero_valoraciones"]
                +
                m
            )
        )
        *
        media_global
    )


    recomendaciones = recomendaciones.merge(
        estadisticas_peliculas,
        on="movieId",
        how="left"
    )


    # -----------------------------------------------------
    # INFORMACIÓN DE PELÍCULAS
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
    # PARTE DE CONTENIDO
    # -----------------------------------------------------

    recomendaciones["score_contenido"] = (
        recomendaciones["genres"]
        .apply(
            lambda generos:
            calcular_score_contenido(
                generos_pelicula=generos,
                perfil_contenido=perfil_contenido
            )
        )
    )


    # -----------------------------------------------------
    # SCORE FINAL
    # -----------------------------------------------------

    recomendaciones["score_final"] = (
        peso_colaborativo
        *
        recomendaciones["score_colaborativo"]
        +
        peso_contenido
        *
        recomendaciones["score_contenido"]
        +
        peso_global
        *
        recomendaciones["rating_regularizado"]
    )


    # -----------------------------------------------------
    # TOP-N
    # -----------------------------------------------------

    recomendaciones = (
        recomendaciones
        .sort_values(
            by=[
                "score_final",
                "numero_vecinos"
            ],
            ascending=[
                False,
                False
            ]
        )
        .head(n_recomendaciones)
        .reset_index(drop=True)
    )


    return recomendaciones[
        [
            "movieId",
            "title",
            "genres",
            "prediccion_vecinos",
            "numero_vecinos",
            "score_colaborativo",
            "score_contenido",
            "rating_regularizado",
            "score_final"
        ]
    ]