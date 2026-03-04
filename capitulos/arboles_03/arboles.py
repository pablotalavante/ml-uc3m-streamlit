import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib.colors import ListedColormap

def render():
    st.header("🌳 Capítulo 3: Árboles de Decisión")
    st.markdown("---")

    # ==========================================
    # SECCIÓN TEÓRICA (Módulos Markdown)
    # ==========================================
    
    st.markdown("""
    ### 1. ¿Qué es un Árbol de Decisión?
    Los árboles de decisión son algoritmos de Machine Learning muy versátiles que pueden realizar tareas de clasificación, regresión e incluso tareas de múltiples salidas. Son modelos muy potentes capaces de ajustarse a conjuntos de datos complejos y constituyen los componentes fundamentales de algoritmos aún más avanzados como los Random Forests (Bosques Aleatorios).

    Se les conoce como modelos de **"caja blanca"** (white box) porque sus decisiones son bastante intuitivas y fáciles de interpretar. A diferencia de otros modelos más opacos, los árboles de decisión proporcionan reglas de clasificación simples que hasta podrían aplicarse manualmente si fuera necesario.
    """)

    st.markdown("""
    ### 2. Estructura y Funcionamiento
    Para hacer una predicción, el árbol se recorre de arriba hacia abajo:
    * **Nodo raíz:** Es el punto de partida (profundidad 0, en la parte superior) donde se hace la primera pregunta sobre uno de los atributos o características del dato.
    * **Nodos intermedios:** Dependiendo de si la respuesta es verdadera o falsa, nos movemos hacia la rama izquierda o derecha, llegando a otros nodos que seguirán haciendo preguntas sobre los atributos.
    * **Nodo hoja:** El recorrido termina cuando llegamos a un nodo hoja. Este tipo de nodo no tiene "hijos" (no hace más preguntas) y su función es simplemente devolver la clase predicha. Además, los árboles pueden estimar la probabilidad de que una instancia pertenezca a una clase concreta calculando la proporción de instancias de esa clase presentes en su nodo hoja.
    """)

    st.markdown("""
    ### 3. ¿Cómo se evalúa la calidad de una división? (Entropía y Gini)
    Para elegir el mejor punto de corte en los datos, los algoritmos miden la "impureza" de las particiones creadas:
    * **Impureza Gini:** Un nodo se considera totalmente "puro" (gini = 0) si todas las instancias de entrenamiento que le aplican pertenecen exactamente a la misma clase.
    * **Entropía y Ganancia de Información:** La entropía es una medida que nos dice cuán lejana está una partición de la perfección o la homogeneidad. A mayor entropía, peor es la partición. La métrica de Ganancia de Información es la diferencia entre la entropía original y la entropía tras aplicar el atributo; por lo que el objetivo del algoritmo es maximizar esta ganancia (es decir, minimizar la entropía).
    """)

    st.markdown("""
    ### 4. Ventajas principales
    Una de las grandes cualidades de los árboles de decisión es que **requieren muy poca preparación de los datos**. En particular, no necesitan que realices procesos de escalado o centrado de las características (como sí requieren otros algoritmos que hemos visto).
    """)

    st.markdown("---")

    # ==========================================
    # SECCIÓN INTERACTIVA
    # ==========================================
    st.subheader("🕹️ Entorno Interactivo")
    
    st.markdown("""
    #### 📊 Sobre los datos de este laboratorio
    Para ilustrar cómo funcionan las fronteras de decisión, estamos utilizando un conjunto de datos sintético clásico llamado **"Make Moons"** (Lunas). 
    Consiste en dos semicírculos de puntos entrelazados (Clase 0 en zonas rojizas y Clase 1 en zonas azuladas). Al añadirle "ruido", los puntos se mezclan en el centro, simulando la incertidumbre de un problema real y obligando al árbol a esforzarse para separarlos.
    
    *Experimenta con los hiperparámetros y observa en la pestaña de **Inspección de un Punto** cómo el modelo toma decisiones paso a paso.*
    """)

    # Configuración de hiperparámetros en columnas
    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### ⚙️ Hiperparámetros")
        
        criterio = st.selectbox(
            "Criterio de Impureza",
            options=["gini", "entropy"]
        )
        
        max_depth = st.slider(
            "Profundidad Máxima",
            min_value=1, max_value=15, value=3,
            help="Define cuántas preguntas sucesivas puede hacer el árbol."
        )
        
        min_samples_leaf = st.slider(
            "Muestras Mínimas por Hoja",
            min_value=1, max_value=20, value=1,
            help="Evita que se creen nodos hoja con muy pocos datos."
        )
        
        ruido = st.slider(
            "Ruido en los datos", 
            min_value=0.0, max_value=0.5, value=0.2,
            help="A mayor ruido, más se mezclan las 'lunas'."
        )
                          
        st.markdown("### 🎲 Datos")
        semilla = st.number_input(
            "Semilla Aleatoria (Seed)", 
            min_value=1, max_value=9999, value=42,
            help="Cambia este valor para generar posiciones de puntos completamente nuevas."
        )

    with col2:
        # Generar y dividir datos
        X, y = make_moons(n_samples=300, noise=ruido, random_state=semilla)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=semilla)

        # Entrenar modelo
        clf = DecisionTreeClassifier(
            criterion=criterio,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=semilla
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Panel de Métricas
        st.markdown("#### 📈 Rendimiento en Datos Nuevos (Test Set)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Exactitud (Accuracy)", f"{accuracy_score(y_test, y_pred):.2f}")
        m2.metric("Precisión", f"{precision_score(y_test, y_pred):.2f}")
        m3.metric("Sensibilidad (Recall)", f"{recall_score(y_test, y_pred):.2f}")
        m4.metric("F1-Score", f"{f1_score(y_test, y_pred):.2f}")
        
        with st.expander("ℹ️ ¿Qué significan estas métricas?"):
            st.markdown("""
            Estas métricas evalúan cómo se comporta el modelo frente a los **datos de prueba**:
            * **Exactitud (Accuracy):** Porcentaje de aciertos totales.
            * **Precisión:** De los etiquetados como Clase 1, ¿cuántos eran realmente Clase 1? (Penaliza Falsos Positivos).
            * **Sensibilidad (Recall):** De los que realmente son Clase 1, ¿cuántos encontró el modelo? (Penaliza Falsos Negativos).
            * **F1-Score:** Equilibrio entre Precisión y Sensibilidad.
            """)
        
        st.markdown("<br>", unsafe_allow_html=True)

        # Pestañas de visualización: Fronteras y Análisis de Camino de Decisión
        tab_fronteras, tab_inspeccion = st.tabs(["🗺️ Fronteras de Decisión", "🔍 Inspección de un Punto (Paso a Paso)"])

        with tab_fronteras:
            fig, ax = plt.subplots(figsize=(8, 5))
            
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
            
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            
            cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
            cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
            
            ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=40, label="Entrenamiento")
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, edgecolor='black', s=60, marker='*', label="Prueba")
            
            ax.set_title(f"Fronteras (Profundidad = {max_depth})")
            ax.set_xlabel("Característica 1 (Eje X)")
            ax.set_ylabel("Característica 2 (Eje Y)")
            ax.legend(loc="best")
            
            st.pyplot(fig)
            plt.close()

        with tab_inspeccion:
            st.markdown("### 📍 Coloca un punto en el mapa")
            st.markdown("Usa los controles para mover el punto amarillo (estrella) y observa cómo el árbol decide a qué clase pertenece.")
            
            cx1, cx2 = st.columns(2)
            with cx1:
                px = st.slider("Posición Característica 1 (X)", float(x_min), float(x_max), 0.0)
            with cx2:
                py = st.slider("Posición Característica 2 (Y)", float(y_min), float(y_max), 0.0)

            # Dibujar el mapa con el punto seleccionado
            fig_punto, ax_punto = plt.subplots(figsize=(8, 3.5))
            ax_punto.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.5) 
            ax_punto.scatter(px, py, c='yellow', edgecolor='black', s=300, marker='*', zorder=5, label="Tu Punto")
            ax_punto.set_xlabel("Característica 1")
            ax_punto.set_ylabel("Característica 2")
            ax_punto.legend(loc="best")
            st.pyplot(fig_punto)
            plt.close()

            # Extraer y mostrar el camino de decisión
            punto_array = np.array([[px, py]])
            camino = clf.decision_path(punto_array).indices
            
            st.markdown("### 🧠 Lógica del Modelo para este punto:")
            
            for nodo in camino:
                # Si el nodo no es una hoja (tiene hijos)
                if clf.tree_.children_left[nodo] != clf.tree_.children_right[nodo]:
                    atributo = clf.tree_.feature[nodo]
                    umbral = clf.tree_.threshold[nodo]
                    valor_punto = punto_array[0, atributo]
                    
                    nombre_attr = "Característica 1 (X)" if atributo == 0 else "Característica 2 (Y)"
                    
                    if valor_punto <= umbral:
                        st.info(f"**Paso:** ¿Es {nombre_attr} ({valor_punto:.2f}) $\le$ {umbral:.2f}? **Sí** ➡️ (Va por la izquierda)")
                    else:
                        st.warning(f"**Paso:** ¿Es {nombre_attr} ({valor_punto:.2f}) $\le$ {umbral:.2f}? **No** ➡️ (Va por la derecha)")
                else:
                    # Es un nodo hoja
                    prediccion_final = clf.classes_[np.argmax(clf.tree_.value[nodo])]
                    color_clase = "roja" if prediccion_final == 0 else "azul"
                    st.success(f"🎯 **Fin del recorrido:** Llegamos a una hoja. El modelo predice que es de la **Clase {prediccion_final}** (zona {color_clase}).")


    # ==========================================
    # SECCIÓN DESPLEGABLE (Deep Dive Gini vs Entropía)
    # ==========================================
    st.markdown("---")
    with st.expander("🔬 ¿Quieres saber más sobre la Impureza Gini y la Entropía?"):
        st.markdown("""
        Tanto **Gini** como **Entropía** son funciones matemáticas que el algoritmo utiliza para evaluar qué tan buena es una división. El objetivo del árbol es siempre dividir los datos de forma que los nodos resultantes sean lo más "puros" posibles (es decir, que contengan datos de una sola clase).

        #### 1. Impureza Gini
        Mide la probabilidad de clasificar incorrectamente un elemento elegido al azar si lo etiquetamos aleatoriamente según la distribución de clases en el nodo.
        * **Fórmula:** $G = 1 - \sum (p_i)^2$ (donde $p_i$ es la proporción de la clase $i$ en el nodo).
        * **Rango:** Va de 0 (nodo perfectamente puro) a 0.5 (nodo totalmente mezclado en un problema binario).

        #### 2. Entropía (Ganancia de Información)
        La entropía es un concepto que viene de la teoría de la información y mide el nivel de "desorden" o incertidumbre en un nodo.
        * **Fórmula:** $H = - \sum p_i \log_2(p_i)$
        * **Rango:** Va de 0 (nodo puro) a 1 (nodo totalmente mezclado en un problema binario).
        """)

    # ==========================================
    # SECCIÓN DE CONCLUSIONES
    # ==========================================
    st.markdown("---")
    st.header("📌 Conclusiones del Capítulo")
    
    st.markdown("""
    A través de este laboratorio interactivo, hemos podido extraer tres grandes lecciones sobre el comportamiento de los Árboles de Decisión:

    1. **Son cajas blancas muy intuitivas:** Como has visto en la inspección paso a paso, el modelo no hace magia matemática indescifrable; simplemente crea un "embudo" de preguntas de *Sí o No* basadas en cortes rectos horizontales y verticales. Esto los hace ideales cuando necesitas explicar y justificar tus predicciones ante usuarios no técnicos.
    2. **El peligro mortal del Sobreajuste (Overfitting):** Si subes la *Profundidad Máxima* a 10 o 15, verás que el árbol empieza a dibujar "islas cuadradas" minúsculas para atrapar puntos individuales de ruido. Aunque el modelo parezca perfecto en el mapa de entrenamiento, las métricas en los datos de prueba caerán. ¡Ha memorizado los datos de memoria en lugar de aprender el concepto general!
    3. **Tienen una alta varianza:** Si mantienes todos los parámetros iguales y solo cambias la *Semilla Aleatoria*, verás cómo las fronteras de decisión cambian radicalmente. Esto demuestra que los árboles de decisión individuales son muy inestables y sensibles a los datos con los que se entrenan (un problema que se soluciona usando **Bosques Aleatorios**).
    """)
