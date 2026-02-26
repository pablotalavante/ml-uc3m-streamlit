import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import seaborn as sns

def render():
    st.header("🔧 Capítulo 7: Preprocesamiento de Datos")
    st.markdown("---")

    st.markdown("""
    El preprocesamiento es una etapa crucial en cualquier proyecto de Machine Learning.
    En este capítulo aprenderás de forma interactiva sobre:
    - **Imputación de valores faltantes**: Cómo rellenar datos perdidos
    - **Escalado de características**: Cómo normalizar los datos para mejorar el rendimiento
    """)

    st.markdown("---")

    # Tabs para las diferentes secciones
    tab1, tab2 = st.tabs(["📊 Imputación de Valores Faltantes", "⚖️ Escalado de Datos"])

    with tab1:
        render_imputacion()

    with tab2:
        render_escalado()


def render_imputacion():
    st.subheader("📊 Imputación de Valores Faltantes")

    st.markdown("""
    Cuando trabajamos con datos reales, es común encontrar **valores faltantes**.
    La imputación es el proceso de rellenar estos valores usando diferentes estrategias.
    """)

    # Configuración de parámetros
    col1, col2 = st.columns(2)

    with col1:
        n_puntos = st.slider("Número de puntos totales", 50, 200, 100, 10)
        porcentaje_faltante = st.slider("Porcentaje de valores faltantes (%)", 10, 50, 20, 5)

    with col2:
        metodo_imputacion = st.selectbox(
            "Método de imputación",
            ["Media", "Mediana", "Moda", "KNN (k=3)", "KNN (k=5)", "Iterativo (MICE)"]
        )

        seed = st.number_input("Semilla aleatoria (para reproducibilidad)", 0, 1000, 42, 1)

    # Generar datos sintéticos
    np.random.seed(seed)

    # Crear dos clusters de puntos
    cluster1 = np.random.randn(n_puntos // 2, 2) + np.array([2, 2])
    cluster2 = np.random.randn(n_puntos // 2, 2) + np.array([-2, -2])
    datos_completos = np.vstack([cluster1, cluster2])

    # Crear versión con valores faltantes
    datos_con_faltantes = datos_completos.copy()
    n_faltantes = int(n_puntos * porcentaje_faltante / 100)

    # Introducir valores faltantes aleatoriamente
    indices_faltantes = np.random.choice(n_puntos, n_faltantes, replace=False)
    for idx in indices_faltantes:
        # Aleatoriamente elegir si falta X, Y, o ambos
        if np.random.rand() > 0.5:
            datos_con_faltantes[idx, 0] = np.nan
        else:
            datos_con_faltantes[idx, 1] = np.nan

    # Aplicar imputación según el método seleccionado
    if metodo_imputacion == "Media":
        imputer = SimpleImputer(strategy='mean')
        datos_imputados = imputer.fit_transform(datos_con_faltantes)

    elif metodo_imputacion == "Mediana":
        imputer = SimpleImputer(strategy='median')
        datos_imputados = imputer.fit_transform(datos_con_faltantes)

    elif metodo_imputacion == "Moda":
        imputer = SimpleImputer(strategy='most_frequent')
        datos_imputados = imputer.fit_transform(datos_con_faltantes)

    elif metodo_imputacion == "KNN (k=3)":
        imputer = KNNImputer(n_neighbors=3)
        datos_imputados = imputer.fit_transform(datos_con_faltantes)

    elif metodo_imputacion == "KNN (k=5)":
        imputer = KNNImputer(n_neighbors=5)
        datos_imputados = imputer.fit_transform(datos_con_faltantes)

    else:  # Iterativo (MICE)
        imputer = IterativeImputer(random_state=seed, max_iter=10)
        datos_imputados = imputer.fit_transform(datos_con_faltantes)

    # Identificar qué puntos fueron imputados
    mascara_faltantes = np.isnan(datos_con_faltantes).any(axis=1)

    # Visualización
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Subplot 1: Datos originales completos
    axes[0].scatter(datos_completos[:, 0], datos_completos[:, 1],
                   alpha=0.6, s=50, c='blue', edgecolors='black', linewidth=0.5)
    axes[0].set_title('Datos Originales (Completos)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Feature X')
    axes[0].set_ylabel('Feature Y')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')

    # Subplot 2: Datos con valores faltantes
    datos_sin_nan = datos_con_faltantes[~mascara_faltantes]
    axes[1].scatter(datos_sin_nan[:, 0], datos_sin_nan[:, 1],
                   alpha=0.6, s=50, c='blue', edgecolors='black', linewidth=0.5,
                   label='Datos completos')
    axes[1].scatter([], [], alpha=0.6, s=50, c='red', marker='x',
                   label=f'Datos faltantes ({n_faltantes})')
    axes[1].set_title(f'Datos con {porcentaje_faltante}% Faltantes', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Feature X')
    axes[1].set_ylabel('Feature Y')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_aspect('equal')

    # Subplot 3: Datos después de imputación
    datos_no_imputados = datos_imputados[~mascara_faltantes]
    datos_imputados_solo = datos_imputados[mascara_faltantes]

    axes[2].scatter(datos_no_imputados[:, 0], datos_no_imputados[:, 1],
                   alpha=0.6, s=50, c='blue', edgecolors='black', linewidth=0.5,
                   label='Datos originales')
    axes[2].scatter(datos_imputados_solo[:, 0], datos_imputados_solo[:, 1],
                   alpha=0.8, s=80, c='red', marker='s', edgecolors='black', linewidth=1,
                   label='Datos imputados')

    # Dibujar líneas desde los puntos originales a los imputados (si es posible)
    for idx in indices_faltantes:
        if not np.all(np.isnan(datos_con_faltantes[idx])):
            axes[2].plot([datos_completos[idx, 0], datos_imputados[idx, 0]],
                        [datos_completos[idx, 1], datos_imputados[idx, 1]],
                        'gray', alpha=0.3, linestyle='--', linewidth=0.5)

    axes[2].set_title(f'Después de Imputación ({metodo_imputacion})', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Feature X')
    axes[2].set_ylabel('Feature Y')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_aspect('equal')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Métricas de error
    st.markdown("### 📏 Evaluación de la Imputación")

    # Calcular error solo en los puntos que fueron imputados
    puntos_originales = datos_completos[mascara_faltantes]
    puntos_imputados = datos_imputados[mascara_faltantes]

    error_mae = np.mean(np.abs(puntos_originales - puntos_imputados))
    error_rmse = np.sqrt(np.mean((puntos_originales - puntos_imputados)**2))

    col1, col2, col3 = st.columns(3)
    col1.metric("Puntos imputados", f"{n_faltantes}")
    col2.metric("Error Absoluto Medio (MAE)", f"{error_mae:.3f}")
    col3.metric("Error Cuadrático Medio (RMSE)", f"{error_rmse:.3f}")

    # Explicación de los métodos
    st.markdown("### 📖 Explicación de los Métodos")

    with st.expander("ℹ️ Ver detalles de cada método"):
        st.markdown("""
        **Media**: Reemplaza valores faltantes con la media de la característica.
        - ✅ Simple y rápido
        - ❌ No considera relaciones entre características

        **Mediana**: Reemplaza valores faltantes con la mediana de la característica.
        - ✅ Más robusto ante outliers que la media
        - ❌ No considera relaciones entre características

        **Moda**: Reemplaza valores faltantes con el valor más frecuente.
        - ✅ Útil para datos categóricos
        - ❌ Menos útil para datos continuos

        **KNN (K-Nearest Neighbors)**: Usa los k vecinos más cercanos para imputar.
        - ✅ Considera relaciones entre características
        - ✅ Mejor preservación de la estructura de datos
        - ❌ Más costoso computacionalmente

        **Iterativo (MICE)**: Modela cada característica con faltantes como función de otras.
        - ✅ Más sofisticado, considera todas las relaciones
        - ✅ Generalmente más preciso
        - ❌ Más lento y complejo
        """)


def render_escalado():
    st.subheader("⚖️ Escalado de Datos")

    st.markdown("""
    El **escalado de características** es crucial para muchos algoritmos de ML que son sensibles
    a la magnitud de las features (como KNN, SVM, redes neuronales).
    Aquí verás cómo diferentes métodos de escalado transforman tus datos.
    """)

    # Configuración
    col1, col2, col3 = st.columns(3)

    with col1:
        n_puntos = st.slider("Número de puntos", 50, 200, 100, 10, key="escalado_n")
        seed = st.number_input("Semilla aleatoria", 0, 1000, 42, 1, key="escalado_seed")

    with col2:
        distribucion = st.selectbox(
            "Tipo de distribución",
            ["Gaussiana", "Uniforme", "Escalas Diferentes"]
        )

    with col3:
        st.markdown("**Añadir outliers:**")
        add_outliers = st.checkbox("Outliers normales", value=False, key="add_outliers")
        add_outliers_extremos = st.checkbox("Outliers extremos", value=False, key="add_outliers_extremos")

    # Generar datos según la distribución seleccionada
    np.random.seed(seed)

    if distribucion == "Gaussiana":
        datos = np.random.randn(n_puntos, 2) * 2 + 5

    elif distribucion == "Uniforme":
        datos = np.random.uniform(0, 10, (n_puntos, 2))

    else:  # Escalas Diferentes
        datos = np.column_stack([
            np.random.randn(n_puntos) * 2 + 5,      # Feature 1: media=5, std=2
            np.random.randn(n_puntos) * 20 + 100    # Feature 2: media=100, std=20
        ])

    # Guardar índices de datos normales
    n_datos_normales = len(datos)
    indices_normales = np.arange(n_datos_normales)
    indices_outliers = np.array([], dtype=int)
    indices_outliers_extremos = np.array([], dtype=int)

    # Agregar outliers normales si está activado
    if add_outliers:
        n_outliers = max(5, n_puntos // 15)  # Al menos 5 outliers
        if distribucion == "Escalas Diferentes":
            # Para escalas diferentes, mantener la proporción
            outliers = np.column_stack([
                np.random.uniform(15, 25, n_outliers),      # Outliers en X
                np.random.uniform(200, 300, n_outliers)     # Outliers en Y
            ])
        else:
            # Para otras distribuciones
            mean = datos.mean(axis=0)
            std = datos.std(axis=0)
            outliers = mean + np.random.randn(n_outliers, 2) * std * 4  # 4 desviaciones estándar

        indices_outliers = np.arange(len(datos), len(datos) + n_outliers)
        datos = np.vstack([datos, outliers])

    # Agregar outliers extremos si está activado
    if add_outliers_extremos:
        n_outliers_ext = max(3, n_puntos // 20)  # Al menos 3 outliers extremos
        if distribucion == "Escalas Diferentes":
            outliers_ext = np.column_stack([
                np.random.uniform(40, 60, n_outliers_ext),      # Outliers extremos en X
                np.random.uniform(500, 700, n_outliers_ext)     # Outliers extremos en Y
            ])
        else:
            mean = datos[:n_datos_normales].mean(axis=0)
            std = datos[:n_datos_normales].std(axis=0)
            outliers_ext = mean + np.random.randn(n_outliers_ext, 2) * std * 8  # 8 desviaciones estándar

        indices_outliers_extremos = np.arange(len(datos), len(datos) + n_outliers_ext)
        datos = np.vstack([datos, outliers_ext])

    # Aplicar diferentes escaladores
    scaler_standard = StandardScaler()
    scaler_minmax = MinMaxScaler()
    scaler_robust = RobustScaler()
    scaler_maxabs = MaxAbsScaler()

    datos_standard = scaler_standard.fit_transform(datos)
    datos_minmax = scaler_minmax.fit_transform(datos)
    datos_robust = scaler_robust.fit_transform(datos)
    datos_maxabs = scaler_maxabs.fit_transform(datos)

    # Crear visualización
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Configurar todos los subplots
    datos_escalados = [
        (datos, "Datos Originales", 0, 0),
        (datos_standard, "StandardScaler\n(z-score)", 0, 1),
        (datos_minmax, "MinMaxScaler\n[0, 1]", 0, 2),
        (datos_robust, "RobustScaler\n(Mediana/IQR)", 1, 0),
        (datos_maxabs, "MaxAbsScaler\n[-1, 1]", 1, 1),
    ]

    for datos_plot, titulo, row, col in datos_escalados:
        ax = axes[row, col]

        # Scatter plot - separar por tipo de punto
        # Datos normales
        ax.scatter(datos_plot[indices_normales, 0], datos_plot[indices_normales, 1],
                  alpha=0.6, s=50, c='blue', edgecolors='black', linewidth=0.5,
                  label='Datos normales')

        # Outliers normales
        if len(indices_outliers) > 0:
            ax.scatter(datos_plot[indices_outliers, 0], datos_plot[indices_outliers, 1],
                      alpha=0.8, s=80, c='orange', marker='^', edgecolors='black', linewidth=1,
                      label='Outliers normales')

        # Outliers extremos
        if len(indices_outliers_extremos) > 0:
            ax.scatter(datos_plot[indices_outliers_extremos, 0], datos_plot[indices_outliers_extremos, 1],
                      alpha=0.9, s=100, c='red', marker='D', edgecolors='black', linewidth=1.5,
                      label='Outliers extremos')

        ax.set_title(titulo, fontsize=14, fontweight='bold')
        ax.set_xlabel('Feature X')
        ax.set_ylabel('Feature Y')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)

        # Agregar leyenda solo si hay outliers
        if len(indices_outliers) > 0 or len(indices_outliers_extremos) > 0:
            ax.legend(fontsize=8, loc='upper right')

        # Agregar estadísticas
        mean_x, mean_y = datos_plot[:, 0].mean(), datos_plot[:, 1].mean()
        std_x, std_y = datos_plot[:, 0].std(), datos_plot[:, 1].std()

        stats_text = f'X: μ={mean_x:.2f}, σ={std_x:.2f}\nY: μ={mean_y:.2f}, σ={std_y:.2f}'
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Ocultar el subplot vacío
    axes[1, 2].axis('off')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Tabla comparativa de estadísticas
    st.markdown("### 📊 Comparación de Estadísticas")

    stats_df = pd.DataFrame({
        'Método': ['Original', 'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler'],
        'Media X': [
            datos[:, 0].mean(),
            datos_standard[:, 0].mean(),
            datos_minmax[:, 0].mean(),
            datos_robust[:, 0].mean(),
            datos_maxabs[:, 0].mean()
        ],
        'Std X': [
            datos[:, 0].std(),
            datos_standard[:, 0].std(),
            datos_minmax[:, 0].std(),
            datos_robust[:, 0].std(),
            datos_maxabs[:, 0].std()
        ],
        'Min X': [
            datos[:, 0].min(),
            datos_standard[:, 0].min(),
            datos_minmax[:, 0].min(),
            datos_robust[:, 0].min(),
            datos_maxabs[:, 0].min()
        ],
        'Max X': [
            datos[:, 0].max(),
            datos_standard[:, 0].max(),
            datos_minmax[:, 0].max(),
            datos_robust[:, 0].max(),
            datos_maxabs[:, 0].max()
        ]
    })

    st.dataframe(stats_df.style.format({
        'Media X': '{:.3f}',
        'Std X': '{:.3f}',
        'Min X': '{:.3f}',
        'Max X': '{:.3f}'
    }), use_container_width=True)

    # Información sobre outliers
    if add_outliers or add_outliers_extremos:
        st.markdown("### 🎯 Impacto de los Outliers")

        col1, col2, col3 = st.columns(3)
        col1.metric("Datos normales", f"{n_datos_normales}")
        col2.metric("Outliers normales", f"{len(indices_outliers)}")
        col3.metric("Outliers extremos", f"{len(indices_outliers_extremos)}")

        # Comparar robustez de escaladores
        st.markdown("**Robustez ante outliers (Rango de valores en X):**")

        robustez_df = pd.DataFrame({
            'Método': ['Original', 'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler'],
            'Rango': [
                datos[:, 0].max() - datos[:, 0].min(),
                datos_standard[:, 0].max() - datos_standard[:, 0].min(),
                datos_minmax[:, 0].max() - datos_minmax[:, 0].min(),
                datos_robust[:, 0].max() - datos_robust[:, 0].min(),
                datos_maxabs[:, 0].max() - datos_maxabs[:, 0].min()
            ]
        })

        st.dataframe(robustez_df.style.format({'Rango': '{:.3f}'}), use_container_width=True)

        st.info("""
        💡 **Observa cómo los outliers afectan cada método:**
        - **StandardScaler y MinMaxScaler**: Muy sensibles a outliers (rango grande)
        - **RobustScaler**: Más robusto, usa mediana e IQR (menos afectado)
        - Los outliers extremos comprimen los datos normales en escaladores no robustos
        """)

    # Explicación de los métodos
    st.markdown("### 📖 Explicación de los Métodos de Escalado")

    with st.expander("ℹ️ Ver detalles de cada método"):
        st.markdown("""
        **StandardScaler (Normalización Z-score)**
        - Fórmula: `(x - μ) / σ`
        - Resultado: Media = 0, Desviación estándar = 1
        - ✅ Útil cuando los datos siguen distribución normal
        - ✅ Requerido para muchos algoritmos (SVM, redes neuronales)
        - ❌ Sensible a outliers

        **MinMaxScaler (Normalización Min-Max)**
        - Fórmula: `(x - min) / (max - min)`
        - Resultado: Valores en rango [0, 1]
        - ✅ Preserva la forma de la distribución original
        - ✅ Útil cuando necesitas valores en rango específico
        - ❌ Muy sensible a outliers

        **RobustScaler**
        - Fórmula: `(x - mediana) / IQR`
        - Usa mediana y rango intercuartílico (IQR)
        - ✅ Robusto ante outliers
        - ✅ Mejor para datos con outliers
        - ❌ No garantiza rango específico

        **MaxAbsScaler**
        - Fórmula: `x / max(|x|)`
        - Resultado: Valores en rango [-1, 1]
        - ✅ No desplaza/centra los datos
        - ✅ Útil para datos sparse (matrices dispersas)
        - ❌ Sensible a outliers

        ---

        **¿Cuál usar?**
        - Datos con outliers → **RobustScaler**
        - Distribución normal → **StandardScaler**
        - Necesitas rango [0,1] → **MinMaxScaler**
        - Datos sparse → **MaxAbsScaler**
        - Redes neuronales → **StandardScaler** o **MinMaxScaler**
        - Árboles de decisión → **Ninguno** (no necesitan escalado)
        """)

    # Comparación visual de distribuciones
    st.markdown("### 📈 Distribución de los Datos (Feature X)")

    fig_hist, axes_hist = plt.subplots(1, 5, figsize=(20, 4))

    for idx, (datos_plot, titulo) in enumerate([
        (datos, "Original"),
        (datos_standard, "Standard"),
        (datos_minmax, "MinMax"),
        (datos_robust, "Robust"),
        (datos_maxabs, "MaxAbs")
    ]):
        # Separar datos normales y outliers para visualización
        if len(indices_outliers) > 0 or len(indices_outliers_extremos) > 0:
            # Histograma de datos normales
            axes_hist[idx].hist(datos_plot[indices_normales, 0], bins=30, alpha=0.7,
                              color='blue', edgecolor='black', label='Normales')

            # Histograma de outliers normales
            if len(indices_outliers) > 0:
                axes_hist[idx].hist(datos_plot[indices_outliers, 0], bins=10, alpha=0.7,
                                  color='orange', edgecolor='black', label='Outliers')

            # Histograma de outliers extremos
            if len(indices_outliers_extremos) > 0:
                axes_hist[idx].hist(datos_plot[indices_outliers_extremos, 0], bins=10, alpha=0.7,
                                  color='red', edgecolor='black', label='Extremos')

            axes_hist[idx].legend(fontsize=8)
        else:
            # Sin outliers, histograma normal
            axes_hist[idx].hist(datos_plot[:, 0], bins=30, alpha=0.7,
                              color='blue', edgecolor='black')

        axes_hist[idx].set_title(titulo, fontweight='bold')
        axes_hist[idx].set_xlabel('Valor Feature X')
        axes_hist[idx].set_ylabel('Frecuencia')
        axes_hist[idx].grid(True, alpha=0.3)

        # Añadir líneas verticales para media/mediana
        media = datos_plot[:, 0].mean()
        axes_hist[idx].axvline(media, color='darkblue', linestyle='--',
                              linewidth=2, alpha=0.5, label=f'μ={media:.2f}')

    plt.tight_layout()
    st.pyplot(fig_hist)
    plt.close()

    # Agregar explicación de lo que se observa
    if add_outliers or add_outliers_extremos:
        st.info("""
        💡 **Observa las distribuciones:**
        - **MinMaxScaler**: Los datos normales se comprimen en un rango pequeño (pico alto)
          y los outliers aparecen en los extremos [0, 1]
        - **StandardScaler**: Los outliers desplazan la media y aumentan la desviación estándar,
          comprimiendo los datos normales cerca de 0
        - **RobustScaler**: Mantiene mejor la forma de la distribución original porque usa
          mediana e IQR (menos afectado por outliers)
        """)
