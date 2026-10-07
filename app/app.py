import streamlit as st
import pandas as pd
import pickle
import sys
from pathlib import Path
from tensorflow.keras.models import load_model


# ---------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------------------------------------

st.set_page_config(
    page_title="Recomendador de películas",
    page_icon="🎬",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))


# ---------------------------------------------------------
# IMPORTS DEL PROYECTO
# ---------------------------------------------------------

from src.models.factores_latentes import recomendar_factores_latentes
from src.models.factores_latentes_hibrido import recomendar_hibrido
from src.models.deep_learning import recomendar_deep_learning
from src.models.deep_learning_ranking import recomendar_deep_learning_ranking
from src.models.cold_start import recomendar_cold_start

from src.models.filtrado_colaborativo import (
    preparar_filtrado_colaborativo,
    recomendar_filtrado_colaborativo
)


# ---------------------------------------------------------
# ESTILO VISUAL
# ---------------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #f5f7fa;
}

h1, h2, h3 {
    color: #1f2937;
}

.stButton > button[kind="primary"] {
    background-color: #e63946;
    color: white;
    border: 1px solid #e63946;
    border-radius: 8px;
    font-weight: 600;
}

.stButton > button[kind="primary"]:hover {
    background-color: #c92f3b;
    color: white;
    border-color: #c92f3b;
}

.stButton > button[kind="secondary"] {
    background-color: white;
    color: #374151;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    font-weight: 500;
}

.stButton > button[kind="secondary"]:hover {
    background-color: #f3f4f6;
    color: #1f2937;
    border-color: #9ca3af;
}

div[data-baseweb="select"] > div {
    border-radius: 8px;
}

div[data-testid="stAlert"] {
    border-radius: 8px;
}

.stProgress > div > div > div > div {
    background-color: #e63946;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 8px 10px;
    border-radius: 8px;
}

div[data-testid="stMetricValue"] {
    font-size: 1.8rem;
}

hr {
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# FUNCIONES AUXILIARES
# ---------------------------------------------------------

def reiniciar_sesion():

    claves = [
        "continuar",
        "valoraciones_nuevo_usuario",
        "peliculas_mostradas",
        "peliculas_nuevo_usuario",
        "nombre_usuario_actual",
        "tipo_usuario_actual",
        "seguir_valorando"
    ]

    for clave in claves:
        if clave in st.session_state:
            del st.session_state[clave]

    for clave in list(st.session_state.keys()):
        if clave.startswith("rating_nuevo_"):
            del st.session_state[clave]


def mostrar_recomendaciones(
    recomendaciones,
    columna_score=None,
    tipo_score="rating"
):

    if recomendaciones.empty:
        st.warning("No se han podido generar recomendaciones.")
        return

    st.subheader("Tus recomendaciones")

    for posicion, (_, pelicula) in enumerate(
        recomendaciones.head(10).iterrows(),
        start=1
    ):

        # Ranking no necesita mostrar score
        if tipo_score == "ranking":
            col_posicion, col_info = st.columns([1, 8])

        else:
            col_posicion, col_info, col_score = st.columns([1, 5, 3])

        with col_posicion:
            st.markdown(f"### {posicion}")

        with col_info:
            st.markdown(f"**{pelicula['title']}**")
            st.caption(pelicula["genres"])

        if tipo_score != "ranking":

            with col_score:

                score = pelicula[columna_score]

                if tipo_score == "rating":

                    # Solo limitamos lo que se muestra en pantalla.
                    # La predicción original no se modifica.
                    score_mostrado = min(max(score, 0.5), 5.0)

                    st.metric(
                        "Valoración estimada",
                        f"{score_mostrado:.2f} / 5"
                    )

                else:

                    st.metric(
                        "Afinidad",
                        f"{score:.3f}"
                    )

        st.divider()


# ---------------------------------------------------------
# ESTADO DE LA APLICACIÓN
# ---------------------------------------------------------

if "continuar" not in st.session_state:
    st.session_state.continuar = False


# ---------------------------------------------------------
# CARGA DE DATOS
# ---------------------------------------------------------

ratings = pd.read_csv(
    BASE_DIR / "data" / "raw" / "ml-latest-small" / "ratings.csv"
)

movies = pd.read_csv(
    BASE_DIR / "data" / "raw" / "ml-latest-small" / "movies.csv"
)


# ---------------------------------------------------------
# FILTRADO COLABORATIVO
# ---------------------------------------------------------

(
    matriz_cf,
    matriz_cf_normalizada,
    medias_cf,
    similitudes_cf
) = preparar_filtrado_colaborativo(
    ratings=ratings
)


# ---------------------------------------------------------
# FACTORES LATENTES
# ---------------------------------------------------------

with open(
    BASE_DIR / "models" / "factores_latentes.pkl",
    "rb"
) as archivo:
    modelo_fl = pickle.load(archivo)

P = modelo_fl["P"]
Q = modelo_fl["Q"]
bias_usuarios = modelo_fl["bias_usuarios"]
bias_peliculas = modelo_fl["bias_peliculas"]
media_global = modelo_fl["media_global"]
user_to_idx = modelo_fl["user_to_idx"]
movie_to_idx = modelo_fl["movie_to_idx"]


# ---------------------------------------------------------
# DEEP LEARNING
# ---------------------------------------------------------

modelo_dl = load_model(
    BASE_DIR / "models" / "modelo_deep_learning.keras"
)

with open(
    BASE_DIR / "models" / "usuario_a_indice.pkl",
    "rb"
) as archivo:
    usuario_a_indice = pickle.load(archivo)

with open(
    BASE_DIR / "models" / "pelicula_a_indice.pkl",
    "rb"
) as archivo:
    pelicula_a_indice = pickle.load(archivo)

with open(
    BASE_DIR / "models" / "columnas_generos.pkl",
    "rb"
) as archivo:
    columnas_generos = pickle.load(archivo)

preferencias_usuario = pd.read_csv(
    BASE_DIR / "models" / "preferencias_usuario.csv"
)


# ---------------------------------------------------------
# DEEP LEARNING RANKING
# ---------------------------------------------------------

modelo_dl_ranking = load_model(
    BASE_DIR / "models" / "modelo_deep_learning_ranking.keras"
)

with open(
    BASE_DIR / "models" / "usuario_a_indice_ranking.pkl",
    "rb"
) as archivo:
    usuario_a_indice_ranking = pickle.load(archivo)

with open(
    BASE_DIR / "models" / "pelicula_a_indice_ranking.pkl",
    "rb"
) as archivo:
    pelicula_a_indice_ranking = pickle.load(archivo)


# ---------------------------------------------------------
# CABECERA
# ---------------------------------------------------------

st.title("🎬 Recomendador de películas")

st.write(
    "Descubre películas adaptadas a tus gustos mediante "
    "diferentes modelos de recomendación."
)


# =========================================================
# PANTALLA INICIAL
# =========================================================

if not st.session_state.continuar:

    nombre = st.text_input(
        "Introduce tu nombre"
    )

    tipo_usuario = st.radio(
        "Selecciona el tipo de usuario",
        [
            "Usuario existente",
            "Nuevo usuario"
        ]
    )

    if st.button(
        "Continuar",
        type="primary"
    ):

        if nombre.strip() == "":
            st.warning(
                "Introduce tu nombre antes de continuar."
            )

        else:
            st.session_state.continuar = True
            st.session_state.nombre_usuario_actual = nombre.strip()
            st.session_state.tipo_usuario_actual = tipo_usuario
            st.rerun()


# =========================================================
# CONTENIDO PRINCIPAL
# =========================================================

else:

    nombre = st.session_state.get(
        "nombre_usuario_actual",
        ""
    )

    tipo_usuario = st.session_state.get(
        "tipo_usuario_actual",
        "Usuario existente"
    )

    col1, col2 = st.columns([4, 1])

    with col1:
        st.write(
            f"Hola, {nombre}"
        )

    with col2:

        if st.button(
            "Cambiar usuario",
            type="secondary"
        ):
            reiniciar_sesion()
            st.rerun()


    # =====================================================
    # USUARIO EXISTENTE
    # =====================================================

    if tipo_usuario == "Usuario existente":

        st.write(
            "Vamos a utilizar tu historial de valoraciones."
        )

        usuarios = sorted(
            ratings["userId"].unique()
        )

        usuario_id = st.selectbox(
            "Selecciona tu ID de usuario",
            usuarios
        )

        historial = ratings[
            ratings["userId"] == usuario_id
        ]

        st.write(
            f"Este usuario ha valorado {len(historial)} películas."
        )

        historial_peliculas = historial.merge(
            movies,
            on="movieId"
        )

        historial_peliculas = (
            historial_peliculas[
                ["title", "rating"]
            ]
            .sort_values(
                by="rating",
                ascending=False
            )
        )

        with st.expander(
            "Ver historial de valoraciones"
        ):

            st.dataframe(
                historial_peliculas,
                hide_index=True,
                use_container_width=True
            )


        # -------------------------------------------------
        # SELECCIÓN DEL MODELO
        # -------------------------------------------------

        modelo_seleccionado = st.selectbox(
            "Selecciona el modelo de recomendación",
            [
                "Filtrado colaborativo",
                "Factores latentes",
                "Factores latentes + popularidad",
                "Deep Learning",
                "Deep Learning Ranking"
            ]
        )


        # -------------------------------------------------
        # GENERAR RECOMENDACIONES
        # -------------------------------------------------

        if st.button(
            "Generar recomendaciones",
            key="generar_existente",
            type="primary"
        ):

            # Filtrado colaborativo
            if modelo_seleccionado == "Filtrado colaborativo":

                recomendaciones = recomendar_filtrado_colaborativo(
                    usuario_id=usuario_id,
                    historial=historial,
                    movies=movies,
                    matriz_normalizada=matriz_cf_normalizada,
                    media_por_usuario=medias_cf,
                    similitud_usuarios=similitudes_cf,
                    k=10,
                    min_vecinos=3,
                    n=10
                )

                columna_score = "prediccion_cf"
                tipo_score = "rating"


            # Factores latentes
            elif modelo_seleccionado == "Factores latentes":

                recomendaciones = recomendar_factores_latentes(
                    usuario_id=usuario_id,
                    historial=historial,
                    movies=movies,
                    P=P,
                    Q=Q,
                    bias_usuarios=bias_usuarios,
                    bias_peliculas=bias_peliculas,
                    media_global=media_global,
                    user_to_idx=user_to_idx,
                    movie_to_idx=movie_to_idx,
                    n=10
                )

                columna_score = "prediccion"
                tipo_score = "rating"


            # Factores latentes + popularidad
            elif modelo_seleccionado == "Factores latentes + popularidad":

                recomendaciones = recomendar_hibrido(
                    usuario_id=usuario_id,
                    historial=historial,
                    ratings=ratings,
                    movies=movies,
                    P=P,
                    Q=Q,
                    bias_usuarios=bias_usuarios,
                    bias_peliculas=bias_peliculas,
                    media_global=media_global,
                    user_to_idx=user_to_idx,
                    movie_to_idx=movie_to_idx,
                    n=10,
                    alpha=0.20
                )

                columna_score = "score_hibrido"
                tipo_score = "score"


            # Deep Learning
            elif modelo_seleccionado == "Deep Learning":

                recomendaciones = recomendar_deep_learning(
                    usuario_id=usuario_id,
                    historial=historial,
                    movies=movies,
                    modelo=modelo_dl,
                    usuario_a_indice=usuario_a_indice,
                    pelicula_a_indice=pelicula_a_indice,
                    columnas_generos=columnas_generos,
                    preferencias_usuario=preferencias_usuario,
                    n=10
                )

                columna_score = "prediccion_dl"
                tipo_score = "rating"


            # Deep Learning Ranking
            else:

                recomendaciones = recomendar_deep_learning_ranking(
                    usuario_id=usuario_id,
                    historial=historial,
                    movies=movies,
                    modelo=modelo_dl_ranking,
                    usuario_a_indice=usuario_a_indice_ranking,
                    pelicula_a_indice=pelicula_a_indice_ranking,
                    n=10
                )

                columna_score = None
                tipo_score = "ranking"


            mostrar_recomendaciones(
                recomendaciones,
                columna_score,
                tipo_score
            )


    # =====================================================
    # NUEVO USUARIO
    # =====================================================

    else:

        st.write(
            "Necesitamos conocer tus gustos antes de recomendarte películas."
        )

        st.write(
            "Valora al menos 5 películas que hayas visto."
        )

        st.caption(
            "Cinco valoraciones son suficientes para crear un perfil inicial, "
            "aunque puedes valorar más películas para aportar más información."
        )


        # -------------------------------------------------
        # ESTADO DEL NUEVO USUARIO
        # -------------------------------------------------

        if "valoraciones_nuevo_usuario" not in st.session_state:
            st.session_state.valoraciones_nuevo_usuario = {}

        if "peliculas_mostradas" not in st.session_state:
            st.session_state.peliculas_mostradas = set()

        if "seguir_valorando" not in st.session_state:
            st.session_state.seguir_valorando = False


        # -------------------------------------------------
        # SELECCIÓN DE PELÍCULAS PARA VALORAR
        # -------------------------------------------------

        peliculas_populares = (
            ratings
            .groupby("movieId")
            .size()
            .sort_values(ascending=False)
            .head(200)
            .index
        )


        def obtener_nueva_tanda():

            disponibles = movies[
                movies["movieId"].isin(
                    peliculas_populares
                )
                &
                ~movies["movieId"].isin(
                    st.session_state.peliculas_mostradas
                )
            ]

            if len(disponibles) == 0:
                return pd.DataFrame()

            cantidad = min(
                5,
                len(disponibles)
            )

            return (
                disponibles
                .sample(n=cantidad)
                .reset_index(drop=True)
            )


        # Primera tanda
        if "peliculas_nuevo_usuario" not in st.session_state:

            st.session_state.peliculas_nuevo_usuario = (
                obtener_nueva_tanda()
            )

        peliculas_nuevo_usuario = (
            st.session_state.peliculas_nuevo_usuario
        )


        # -------------------------------------------------
        # PROGRESO
        # -------------------------------------------------

        numero_valoraciones = len(
            st.session_state.valoraciones_nuevo_usuario
        )

        st.progress(
            min(
                numero_valoraciones / 5,
                1.0
            )
        )

        st.write(
            f"Valoraciones guardadas: {numero_valoraciones} / 5"
        )


        # -------------------------------------------------
        # PELÍCULAS PARA VALORAR
        # -------------------------------------------------

        if (
            numero_valoraciones < 5
            or st.session_state.seguir_valorando
        ):

            valoraciones_tanda = {}

            for _, pelicula in peliculas_nuevo_usuario.iterrows():

                st.write(
                    f"**{pelicula['title']}**"
                )

                st.caption(
                    pelicula["genres"]
                )

                valoracion = st.selectbox(
                    "Tu valoración",
                    [
                        "No la he visto",
                        0.5,
                        1.0,
                        1.5,
                        2.0,
                        2.5,
                        3.0,
                        3.5,
                        4.0,
                        4.5,
                        5.0
                    ],
                    key=f"rating_nuevo_{pelicula['movieId']}"
                )

                if valoracion != "No la he visto":

                    valoraciones_tanda[
                        int(pelicula["movieId"])
                    ] = float(valoracion)


            if st.button(
                "Guardar valoraciones y mostrar otras",
                type="primary"
            ):

                st.session_state.valoraciones_nuevo_usuario.update(
                    valoraciones_tanda
                )

                st.session_state.peliculas_mostradas.update(
                    peliculas_nuevo_usuario["movieId"].tolist()
                )

                st.session_state.peliculas_nuevo_usuario = (
                    obtener_nueva_tanda()
                )

                if len(
                    st.session_state.valoraciones_nuevo_usuario
                ) >= 5:

                    st.session_state.seguir_valorando = False

                st.rerun()


        # -------------------------------------------------
        # PERFIL INICIAL COMPLETADO
        # -------------------------------------------------

        numero_valoraciones = len(
            st.session_state.valoraciones_nuevo_usuario
        )

        if numero_valoraciones >= 5:

            st.success(
                "Ya tenemos suficientes valoraciones para construir tu perfil."
            )

            st.write(
                f"Has valorado {numero_valoraciones} películas."
            )

            valoraciones_df = pd.DataFrame(
                [
                    {
                        "movieId": movie_id,
                        "rating": rating
                    }
                    for movie_id, rating
                    in st.session_state.valoraciones_nuevo_usuario.items()
                ]
            )

            valoraciones_df = valoraciones_df.merge(
                movies[
                    ["movieId", "title"]
                ],
                on="movieId"
            )


            # -------------------------------------------------
            # GENERAR RECOMENDACIONES COLD-START
            # -------------------------------------------------

            if st.button(
                "Generar recomendaciones",
                key="generar_nuevo",
                type="primary"
            ):

                recomendaciones_cold = recomendar_cold_start(
                    valoraciones=st.session_state.valoraciones_nuevo_usuario,
                    ratings=ratings,
                    movies=movies,
                    media_global=media_global,
                    min_comunes=3,
                    n_vecinos=10,
                    n_recomendaciones=10,
                    min_vecinos=3,
                    peso_colaborativo=0.55,
                    peso_contenido=0.30,
                    peso_global=0.15,
                    m=20,
                    constante_confianza_vecinos=3
                )

                mostrar_recomendaciones(
                    recomendaciones_cold,
                    "score_final",
                    "score"
                )


            # -------------------------------------------------
            # VALORACIONES REGISTRADAS
            # -------------------------------------------------

            with st.expander(
                "Ver valoraciones registradas"
            ):

                st.dataframe(
                    valoraciones_df[
                        ["title", "rating"]
                    ],
                    hide_index=True,
                    use_container_width=True
                )


            # -------------------------------------------------
            # SEGUIR VALORANDO
            # -------------------------------------------------

            if not st.session_state.seguir_valorando:

                if st.button(
                    "Valorar más películas",
                    type="secondary"
                ):

                    st.session_state.seguir_valorando = True
                    st.rerun()


        else:

            faltan = (
                5
                -
                numero_valoraciones
            )

            st.info(
                f"Necesitas valorar {faltan} película(s) más."
            )