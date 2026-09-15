import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Predicción de Resistencia del Concreto",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# ESTILOS DE LA APLICACIÓN
# ============================================================

st.markdown("""
<style>

    /* ---------- FONDO GENERAL ---------- */

    .stApp {
        background-color: #f4f6f8;
    }

    .main .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ---------- BARRA LATERAL ---------- */

    section[data-testid="stSidebar"] {
        background-color: #17212b;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #ffffff !important;
    }


    /* ---------- CABECERA ---------- */

    .header-box {
        background: #ffffff;
        border-left: 7px solid #d97706;
        padding: 25px 30px;
        border-radius: 8px;
        margin-bottom: 25px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }

    .header-title {
        font-size: 30px;
        font-weight: 700;
        color: #17212b;
        margin: 0;
    }

    .header-subtitle {
        font-size: 15px;
        color: #68737d;
        margin-top: 7px;
    }


    /* ---------- SECCIONES ---------- */

    .section-title {
        font-size: 19px;
        font-weight: 700;
        color: #17212b;
        border-bottom: 2px solid #d97706;
        padding-bottom: 8px;
        margin-top: 10px;
        margin-bottom: 18px;
    }


    /* ---------- TARJETAS ---------- */

    .info-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 18px 20px;
        border: 1px solid #e2e6e9;
        box-shadow: 0 2px 7px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }

    .card-title {
        font-size: 13px;
        color: #737d86;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
    }

    .card-value {
        font-size: 24px;
        font-weight: 700;
        color: #17212b;
    }


    /* ---------- RESULTADO ---------- */

    .result-box {
        background: #ffffff;
        border-radius: 10px;
        padding: 28px;
        border: 1px solid #d9dee3;
        border-top: 6px solid #d97706;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
        text-align: center;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .result-label {
        color: #69747e;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .result-value {
        color: #17212b;
        font-size: 44px;
        font-weight: 800;
        margin-top: 8px;
    }

    .result-unit {
        color: #69747e;
        font-size: 17px;
    }


    /* ---------- BOTÓN ---------- */

    div.stButton > button {
        width: 100%;
        height: 52px;
        background-color: #d97706;
        color: white;
        border: none;
        border-radius: 7px;
        font-size: 16px;
        font-weight: 700;
        letter-spacing: 0.3px;
    }

    div.stButton > button:hover {
        background-color: #b45309;
        color: white;
    }


    /* ---------- ALERTAS ---------- */

    .warning-box {
        background-color: #fff7ed;
        border-left: 5px solid #d97706;
        padding: 15px;
        border-radius: 5px;
        color: #7c4511;
    }


    /* ---------- PIE ---------- */

    .footer {
        margin-top: 45px;
        padding-top: 15px;
        border-top: 1px solid #d9dee3;
        text-align: center;
        color: #7b858e;
        font-size: 12px;
    }


    /* ---------- OCULTAR ELEMENTOS DE STREAMLIT ---------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# RUTAS DE ARCHIVOS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODELO_PATH = BASE_DIR / "modelo_gradient_boosting.pkl"
SCALER_PATH = BASE_DIR / "scaler GB.pkl"


# ============================================================
# CARGA DEL MODELO Y SCALER
# ============================================================

@st.cache_resource
def cargar_modelos():

    if not MODELO_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo:\n{MODELO_PATH}"
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el scaler:\n{SCALER_PATH}"
        )

    modelo = joblib.load(MODELO_PATH)
    scaler = joblib.load(SCALER_PATH)

    return modelo, scaler


try:
    modelo, scaler = cargar_modelos()

except Exception as e:

    st.error("No fue posible cargar el modelo o el escalador.")

    st.code(str(e))

    st.stop()


# ============================================================
# CONFIGURACIÓN EXPERIMENTAL
# ============================================================

CONFIGURACION = {

    175: {
        "Manual": {
            "ac": 0.82,
            "fino": 53,
            "grueso": 47
        }
    },

    210: {
        "Manual": {
            "ac": 0.77,
            "fino": 53,
            "grueso": 47
        }
    },

    280: {
        "Manual": {
            "ac": 0.65,
            "fino": 54,
            "grueso": 46
        }
    },

    350: {
        "Manual": {
            "ac": 0.53,
            "fino": 51,
            "grueso": 49
        },

        "Pavimentadora": {
            "ac": 0.51,
            "fino": 45,
            "grueso": 55
        }
    }
}


EDADES = [7, 28]


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def obtener_configuracion(diseno, tipo):

    return CONFIGURACION[diseno][tipo]


def generar_slumps(inicio, fin, paso=0.25):

    valores = np.arange(
        inicio,
        fin + paso / 2,
        paso
    )

    return [round(float(x), 2) for x in valores]


def construir_variables(
    diseno,
    tipo,
    ac,
    fino,
    grueso,
    slump,
    edad
):

    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    inversa_ac = 1 / ac

    edad_2 = edad ** 2

    afino_agueso = fino / grueso

    ac_edad = ac * edad


    # --------------------------------------------------------
    # VARIABLES PRINCIPALES
    # --------------------------------------------------------

    variables = {

        "Diseño": diseno,

        "Relacion Agua/Cemento": ac,

        "Agregado Fino (%)": fino,

        "Agregado Grueso (%)": grueso,

        "Slump (in)": slump,

        "Edad (dias)": edad,

        "Inversa_AC": inversa_ac,

        "Edad^2": edad_2,

        # IMPORTANTE:
        # Este nombre debe coincidir exactamente con
        # el utilizado durante el entrenamiento.
        "Afino/Agueso": afino_agueso,

        "Ac*Edad": ac_edad,


        # ----------------------------------------------------
        # VARIABLES DUMMY DEL DISEÑO
        # ----------------------------------------------------

        "Diseño_175": int(diseno == 175),

        "Diseño_210": int(diseno == 210),

        "Diseño_280": int(diseno == 280),

        "Diseño_350": int(diseno == 350),


        # ----------------------------------------------------
        # VARIABLES DUMMY DEL TIPO DE COLOCACIÓN
        # ----------------------------------------------------

        "Tipo_Colocacion_Manual": int(tipo == "Manual"),

        "Tipo_Colocacion_Pavimentadora": int(
            tipo == "Pavimentadora"
        ),


        # ----------------------------------------------------
        # VARIABLES DE COMBINACIÓN DISEÑO + COLOCACIÓN
        # ----------------------------------------------------

        "Diseño_Final_175_Manual": int(
            diseno == 175 and tipo == "Manual"
        ),

        "Diseño_Final_210_Manual": int(
            diseno == 210 and tipo == "Manual"
        ),

        "Diseño_Final_280_Manual": int(
            diseno == 280 and tipo == "Manual"
        ),

        "Diseño_Final_350_Manual": int(
            diseno == 350 and tipo == "Manual"
        ),

        "Diseño_Final_350_Pavimentadora": int(
            diseno == 350 and tipo == "Pavimentadora"
        )
    }

    return variables


def preparar_entrada(variables):

    # ============================================================
    # NOMBRES EXACTOS UTILIZADOS POR EL SCALER
    # ============================================================

    features_esperadas = list(scaler.feature_names_in_)

    # ============================================================
    # VARIABLES DERIVADAS
    # ============================================================

    variables["Inversa_AC"] = (
        1 / variables["Relacion Agua/Cemento"]
    )

    variables["Edad^2"] = (
        variables["Edad (dias)"] ** 2
    )

    variables["Afino/Agueso"] = (
        variables["Agregado Fino (%)"] /
        variables["Agregado Grueso (%)"]
    )

    variables["Ac*Edad"] = (
        variables["Relacion Agua/Cemento"] *
        variables["Edad (dias)"]
    )

    # ============================================================
    # COMPATIBILIDAD CON LOS NOMBRES DEL SCALER
    # ============================================================

    # El scaler fue entrenado con espacio entre "Tipo" y "Colocacion"
    variables["Tipo Colocacion_Manual"] = (
        int(variables.get("Tipo_Colocacion_Manual", 0) == 1)
    )

    variables["Tipo Colocacion_Pavimentadora"] = (
        int(variables.get("Tipo_Colocacion_Pavimentadora", 0) == 1)
    )

    # ============================================================
    # VERIFICAR VARIABLES FALTANTES
    # ============================================================

    faltantes = [
        variable
        for variable in features_esperadas
        if variable not in variables
    ]

    if faltantes:
        raise ValueError(
            "Faltan variables requeridas por el scaler: "
            f"{faltantes}"
        )

    # ============================================================
    # CREAR ENTRADA EN EL MISMO ORDEN DEL ENTRENAMIENTO
    # ============================================================

    entrada = pd.DataFrame(
        [{
            variable: variables[variable]
            for variable in features_esperadas
        }]
    )

    return entrada, features_esperadas
def realizar_prediccion(
    diseno,
    tipo,
    ac,
    fino,
    grueso,
    slump,
    edad
):

    variables = construir_variables(
        diseno=diseno,
        tipo=tipo,
        ac=ac,
        fino=fino,
        grueso=grueso,
        slump=slump,
        edad=edad
    )

    entrada, features = preparar_entrada(
        variables
    )

    # --------------------------------------------------------
    # ESCALAMIENTO
    # --------------------------------------------------------

    entrada_escalada = scaler.transform(
        entrada
    )

    # --------------------------------------------------------
    # PREDICCIÓN
    # --------------------------------------------------------

    prediccion = modelo.predict(
        entrada_escalada
    )

    return float(prediccion[0]), variables, entrada, features


# ============================================================
# CABECERA
# ============================================================

st.markdown("""
<div class="header-box">

    <div class="header-title">
        Predicción de Resistencia a Compresión del Concreto
    </div>

    <div class="header-subtitle">
        Aplicación basada en un modelo de Machine Learning
        para estimar la resistencia del concreto a partir
        de las condiciones de diseño y ensayo.
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# BARRA LATERAL
# ============================================================

with st.sidebar:

    st.markdown(
        "## PARÁMETROS DE INGRESO"
    )

    st.markdown(
        "---"
    )


    # --------------------------------------------------------
    # DISEÑO
    # --------------------------------------------------------

    diseno = st.selectbox(
        "Diseño del concreto",
        options=[175, 210, 280, 350],
        format_func=lambda x: f"{x} kg/cm²"
    )


    # --------------------------------------------------------
    # TIPO DE COLOCACIÓN
    # --------------------------------------------------------

    tipos_disponibles = list(
        CONFIGURACION[diseno].keys()
    )

    tipo = st.selectbox(
        "Tipo de colocación",
        options=tipos_disponibles
    )


    # --------------------------------------------------------
    # OBTENER PARÁMETROS FIJOS
    # --------------------------------------------------------

    config = obtener_configuracion(
        diseno,
        tipo
    )

    ac = config["ac"]

    fino = config["fino"]

    grueso = config["grueso"]


    # --------------------------------------------------------
    # SLUMP
    # --------------------------------------------------------

    st.markdown(
        "### Consistencia"
    )

    if (
        diseno == 350
        and tipo == "Pavimentadora"
    ):

        slumps = generar_slumps(
            2.0,
            4.0
        )

        slump_default = 3.0

    else:

        slumps = generar_slumps(
            4.5,
            7.5
        )

        slump_default = 6.0


    slump = st.select_slider(
        "Slump",
        options=slumps,
        value=slump_default,
        format_func=lambda x: f"{x:.2f} in"
    )


    # --------------------------------------------------------
    # EDAD
    # --------------------------------------------------------

    edad = st.selectbox(
        "Edad del concreto",
        options=EDADES,
        format_func=lambda x: f"{x} días"
    )


    st.markdown("---")


    st.caption(
        "Los parámetros de relación agua/cemento y "
        "proporciones de agregados son asignados "
        "automáticamente según el diseño seleccionado."
    )


# ============================================================
# CUERPO PRINCIPAL
# ============================================================

st.markdown(
    '<div class="section-title">Condiciones de diseño seleccionadas</div>',
    unsafe_allow_html=True
)


# ============================================================
# TARJETAS DE INFORMACIÓN
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        f"""
        <div class="info-card">
            <div class="card-title">Diseño</div>
            <div class="card-value">{diseno}</div>
            <div style="color:#68737d;">kg/cm²</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="info-card">
            <div class="card-title">A/C</div>
            <div class="card-value">{ac:.2f}</div>
            <div style="color:#68737d;">Relación agua/cemento</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        f"""
        <div class="info-card">
            <div class="card-title">Slump</div>
            <div class="card-value">{slump:.2f}</div>
            <div style="color:#68737d;">pulgadas</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        f"""
        <div class="info-card">
            <div class="card-title">Edad</div>
            <div class="card-value">{edad}</div>
            <div style="color:#68737d;">días</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# INFORMACIÓN DEL MATERIAL
# ============================================================

st.markdown(
    '<div class="section-title">Parámetros de mezcla</div>',
    unsafe_allow_html=True
)


col_a, col_b, col_c = st.columns(3)


with col_a:

    st.metric(
        "Agregado fino",
        f"{fino}%"
    )


with col_b:

    st.metric(
        "Agregado grueso",
        f"{grueso}%"
    )


with col_c:

    st.metric(
        "Colocación",
        tipo
    )


# ============================================================
# BOTÓN DE PREDICCIÓN
# ============================================================

st.markdown("")

_, col_button, _ = st.columns(
    [1, 2, 1]
)


with col_button:

    ejecutar = st.button(
        "ESTIMAR RESISTENCIA",
        use_container_width=True
    )


# ============================================================
# PREDICCIÓN
# ============================================================

if ejecutar:

    try:

        prediccion, variables, entrada, features = realizar_prediccion(
            diseno=diseno,
            tipo=tipo,
            ac=ac,
            fino=fino,
            grueso=grueso,
            slump=slump,
            edad=edad
        )


        # ----------------------------------------------------
        # DIFERENCIA CON EL DISEÑO
        # ----------------------------------------------------

        diferencia = prediccion - diseno

        porcentaje = (
            diferencia / diseno
        ) * 100


        # ----------------------------------------------------
        # RESULTADO PRINCIPAL
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="result-box">

                <div class="result-label">
                    Resistencia estimada a compresión
                </div>

                <div class="result-value">
                    {prediccion:,.2f}
                </div>

                <div class="result-unit">
                    kg/cm²
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # COMPARACIÓN
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Comparación con la resistencia de diseño</div>',
            unsafe_allow_html=True
        )


        c1, c2, c3 = st.columns(3)


        with c1:

            st.metric(
                "Resistencia de diseño",
                f"{diseno:,.2f} kg/cm²"
            )


        with c2:

            st.metric(
                "Resistencia estimada",
                f"{prediccion:,.2f} kg/cm²"
            )


        with c3:

            st.metric(
                "Diferencia",
                f"{diferencia:+,.2f} kg/cm²",
                delta=f"{porcentaje:+.2f}%"
            )


        # ----------------------------------------------------
        # FEATURE ENGINEERING
        # ----------------------------------------------------

        with st.expander(
            "Ver variables derivadas del modelo"
        ):

            st.write(
                "Variables calculadas automáticamente "
                "mediante el proceso de Feature Engineering:"
            )

            feature_engineering = pd.DataFrame({

                "Variable": [
                    "Inversa_AC",
                    "Edad^2",
                    "Afino/Agueso",
                    "Ac*Edad"
                ],

                "Valor": [
                    variables["Inversa_AC"],
                    variables["Edad^2"],
                    variables["Afino/Agueso"],
                    variables["Ac*Edad"]
                ]
            })

            st.dataframe(
                feature_engineering,
                use_container_width=True,
                hide_index=True
            )


        # ----------------------------------------------------
        # VARIABLES UTILIZADAS
        # ----------------------------------------------------

        with st.expander(
            "Ver variables utilizadas por el modelo"
        ):

            tabla_variables = entrada.T.reset_index()

            tabla_variables.columns = [
                "Variable",
                "Valor"
            ]

            st.dataframe(
                tabla_variables,
                use_container_width=True,
                hide_index=True
            )


        # ----------------------------------------------------
        # INFORMACIÓN TÉCNICA
        # ----------------------------------------------------

        with st.expander(
            "Información técnica del modelo"
        ):

            st.write(
                f"**Modelo:** Gradient Boosting Regressor"
            )

            st.write(
                f"**Número de variables de entrada:** "
                f"{len(features)}"
            )

            st.write(
                f"**Variables esperadas por el scaler:**"
            )

            for feature in features:

                st.write(
                    f"- {feature}"
                )


    except Exception as e:

        st.error(
            "No fue posible realizar la predicción."
        )

        st.exception(e)


# ============================================================
# INFORMACIÓN METODOLÓGICA
# ============================================================

st.markdown(
    '<div class="section-title">Consideraciones</div>',
    unsafe_allow_html=True
)

st.info(
    "La resistencia mostrada corresponde a una estimación "
    "realizada mediante el modelo de Machine Learning "
    "entrenado con los datos experimentales. La predicción "
    "no sustituye la verificación mediante ensayos "
    "de laboratorio."
)


# ============================================================
# PIE DE PÁGINA
# ============================================================

st.markdown("""
<div class="footer">

    Sistema de predicción de resistencia del concreto mediante Machine Learning
    <br>
    Proyecto de investigación – Ingeniería Civil

</div>
""", unsafe_allow_html=True)
