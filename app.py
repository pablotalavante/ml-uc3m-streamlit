import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Machine Learning - Tutorial Interactivo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importar módulos de capítulos
from capitulos.introduccion_01 import introduccion
from capitulos.knn_02 import knn
from capitulos.arboles_03 import arboles
from capitulos.evaluaciones_04 import evaluaciones
from capitulos.metricas_05 import metricas
from capitulos.ajuste_hiperparametros_06 import ajuste_hiperparametros
from capitulos.preproceso_07 import preproceso
from capitulos.modelos_lineales_08 import modelos_lineales

def main():
    st.title("🤖 Machine Learning - Tutorial Interactivo")
    st.markdown("---")

    # Sidebar para navegación
    st.sidebar.title("📚 Índice de Contenidos")
    st.sidebar.markdown("---")

    capitulo = st.sidebar.radio(
        "Selecciona un capítulo:",
        [
            "1. Introducción",
            "2. K-Nearest Neighbors (KNN)",
            "3. Árboles de Decisión",
            "4. Evaluación de Modelos",
            "5. Métricas de Rendimiento",
            "6. Ajuste de Hiperparámetros",
            "7. Preprocesamiento de Datos",
            "8. Modelos Lineales"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tip**: Interactúa con los ejemplos para entender mejor los conceptos")

    # Renderizar el capítulo seleccionado
    if capitulo == "1. Introducción":
        introduccion.render()
    elif capitulo == "2. K-Nearest Neighbors (KNN)":
        knn.render()
    elif capitulo == "3. Árboles de Decisión":
        arboles.render()
    elif capitulo == "4. Evaluación de Modelos":
        evaluaciones.render()
    elif capitulo == "5. Métricas de Rendimiento":
        metricas.render()
    elif capitulo == "6. Ajuste de Hiperparámetros":
        ajuste_hiperparametros.render()
    elif capitulo == "7. Preprocesamiento de Datos":
        preproceso.render()
    elif capitulo == "8. Modelos Lineales":
        modelos_lineales.render()

if __name__ == "__main__":
    main()
