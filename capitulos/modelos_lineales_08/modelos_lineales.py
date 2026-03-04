import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import make_classification
import pandas as pd

def render():
    st.header("📊 Capítulo 8: Modelos Lineales")
    st.markdown("---")

    # Selector de sección
    seccion = st.selectbox(
        "Selecciona un tema:",
        [
            "1. Regresión Lineal Simple",
            "2. Regresión Polinómica",
            "3. Gradient Descent Interactivo",
            "4. Regularización (L1, L2, Elastic Net)",
            "5. Regresión Logística",
            "6. Red Neuronal Sencilla"
        ]
    )

    st.markdown("---")

    if seccion == "1. Regresión Lineal Simple":
        render_regresion_lineal_simple()
    elif seccion == "2. Regresión Polinómica":
        render_regresion_polinomica()
    elif seccion == "3. Gradient Descent Interactivo":
        render_gradient_descent()
    elif seccion == "4. Regularización (L1, L2, Elastic Net)":
        render_regularizacion()
    elif seccion == "5. Regresión Logística":
        render_regresion_logistica()
    elif seccion == "6. Red Neuronal Sencilla":
        render_red_neuronal()


def render_regresion_lineal_simple():
    st.subheader("📈 1. Regresión Lineal Simple")

    st.markdown("""
    La regresión lineal simple modela la relación entre una variable independiente $x$ y
    una variable dependiente $y$ mediante una ecuación lineal:

    $$y = w_0 + w_1 x + \\epsilon$$

    donde:
    - $w_0$ es el intercepto (bias)
    - $w_1$ es la pendiente (peso)
    - $\\epsilon$ es el error
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Parámetros del dataset")
        n_samples = st.slider("Número de muestras", 10, 200, 50, key="lr_samples")
        noise = st.slider("Nivel de ruido", 0.0, 50.0, 10.0, key="lr_noise")
        true_slope = st.slider("Pendiente real", -5.0, 5.0, 2.0, key="lr_slope")
        true_intercept = st.slider("Intercepto real", -50.0, 50.0, 10.0, key="lr_intercept")

    # Generar datos
    np.random.seed(42)
    X = np.linspace(0, 10, n_samples)
    y_true = true_slope * X + true_intercept
    y = y_true + np.random.normal(0, noise, n_samples)

    # Entrenar modelo
    X_reshaped = X.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X_reshaped, y)
    y_pred = model.predict(X_reshaped)

    # Calcular métricas
    mse = np.mean((y - y_pred) ** 2)
    r2 = model.score(X_reshaped, y)

    with col2:
        st.markdown("#### Resultados del modelo")
        st.write(f"**Pendiente estimada (w₁):** {model.coef_[0]:.3f}")
        st.write(f"**Intercepto estimado (w₀):** {model.intercept_:.3f}")
        st.write(f"**MSE:** {mse:.3f}")
        st.write(f"**R² Score:** {r2:.3f}")

    # Visualización
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, alpha=0.5, label='Datos observados')
    ax.plot(X, y_true, 'g--', linewidth=2, label='Relación real')
    ax.plot(X, y_pred, 'r-', linewidth=2, label='Predicción del modelo')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('Regresión Lineal Simple')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

    st.markdown("""
    **Interpretación:**
    - Los puntos azules son los datos observados con ruido.
    - La línea verde discontinua representa la relación real (sin ruido).
    - La línea roja es la predicción del modelo de regresión lineal.
    """)


def render_regresion_polinomica():
    st.subheader("🔄 2. Regresión Polinómica")

    st.markdown("""
    La regresión polinómica extiende la regresión lineal usando potencias de la variable independiente:

    $$y = w_0 + w_1 x + w_2 x^2 + w_3 x^3 + ... + w_n x^n$$

    Esto permite modelar relaciones no lineales.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Parámetros de los datos")
        data_complexity = st.slider("Complejidad de los datos (grado real)", 1, 5, 3, key="poly_data_degree")
        n_samples = st.slider("Número de muestras", 20, 200, 80, key="poly_samples")
        noise = st.slider("Nivel de ruido", 0.0, 2.0, 0.5, key="poly_noise")
        test_size = st.slider("Tamaño del conjunto de test (%)", 10, 50, 20, key="poly_test_size") / 100

    with col2:
        st.markdown("#### Parámetros del modelo")
        degree = st.slider("Grado del polinomio del modelo", 1, 15, 3, key="poly_model_degree")
        compare_models = st.checkbox("Comparar múltiples grados", value=False, key="poly_compare")

    # Generar datos no lineales basados en la complejidad elegida
    np.random.seed(42)
    X = np.linspace(-3, 3, n_samples)

    # Generar función real según la complejidad
    if data_complexity == 1:
        y_true = 2 * X + 1
    elif data_complexity == 2:
        y_true = 0.5 * X**2 + X + 1
    elif data_complexity == 3:
        y_true = 0.5 * X**3 - 2 * X**2 + X + 1
    elif data_complexity == 4:
        y_true = 0.1 * X**4 - 0.5 * X**3 + X + 1
    else:  # 5
        y_true = 0.05 * X**5 - 0.2 * X**4 + 0.3 * X**3 + X + 1

    y = y_true + np.random.normal(0, noise, n_samples)

    # Split train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    X_train_reshaped = X_train.reshape(-1, 1)
    X_test_reshaped = X_test.reshape(-1, 1)

    if not compare_models:
        # Entrenar un solo modelo
        poly_features = PolynomialFeatures(degree=degree)
        X_train_poly = poly_features.fit_transform(X_train_reshaped)
        X_test_poly = poly_features.transform(X_test_reshaped)

        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        # Predicciones
        y_train_pred = model.predict(X_train_poly)
        y_test_pred = model.predict(X_test_poly)

        # Predicciones suaves para visualización
        X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
        X_plot_poly = poly_features.transform(X_plot)
        y_plot = model.predict(X_plot_poly)

        # Métricas
        mse_train = np.mean((y_train - y_train_pred) ** 2)
        mse_test = np.mean((y_test - y_test_pred) ** 2)
        r2_train = model.score(X_train_poly, y_train)
        r2_test = model.score(X_test_poly, y_test)

        st.markdown("#### Resultados")
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"**MSE Train:** {mse_train:.3f}")
            st.write(f"**R² Train:** {r2_train:.3f}")
        with col_b:
            st.write(f"**MSE Test:** {mse_test:.3f}")
            st.write(f"**R² Test:** {r2_test:.3f}")

        # Visualización
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(X_train, y_train, alpha=0.6, label='Train', s=50, edgecolors='k')
        ax.scatter(X_test, y_test, alpha=0.6, label='Test', s=50, marker='s', edgecolors='k')
        ax.plot(X, y_true, 'g--', linewidth=2, label='Función real')
        ax.plot(X_plot, y_plot, 'r-', linewidth=3, label=f'Modelo (grado {degree})')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.set_title(f'Regresión Polinómica: Datos grado {data_complexity}, Modelo grado {degree}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    else:
        # Comparar múltiples modelos
        degrees_to_test = [1, 2, 3, 5, 7, 10, 15]
        train_errors = []
        test_errors = []

        models_predictions = {}

        for d in degrees_to_test:
            poly_features = PolynomialFeatures(degree=d)
            X_train_poly = poly_features.fit_transform(X_train_reshaped)
            X_test_poly = poly_features.transform(X_test_reshaped)

            model = LinearRegression()
            model.fit(X_train_poly, y_train)

            y_train_pred = model.predict(X_train_poly)
            y_test_pred = model.predict(X_test_poly)

            train_errors.append(np.mean((y_train - y_train_pred) ** 2))
            test_errors.append(np.mean((y_test - y_test_pred) ** 2))

            # Guardar predicciones para visualización
            X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
            X_plot_poly = poly_features.transform(X_plot)
            models_predictions[d] = model.predict(X_plot_poly)

        st.markdown("#### Comparación de modelos")

        # Crear tabla de resultados
        results_df = pd.DataFrame({
            'Grado': degrees_to_test,
            'MSE Train': [f"{e:.3f}" for e in train_errors],
            'MSE Test': [f"{e:.3f}" for e in test_errors]
        })
        st.dataframe(results_df, use_container_width=True)

        # Visualizaciones
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Gráfico 1: Curva de complejidad
        ax1.plot(degrees_to_test, train_errors, 'o-', linewidth=2, markersize=8, label='Error Train')
        ax1.plot(degrees_to_test, test_errors, 's-', linewidth=2, markersize=8, label='Error Test')
        ax1.set_xlabel('Grado del Polinomio')
        ax1.set_ylabel('MSE')
        ax1.set_title('Curva de Complejidad del Modelo')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(degrees_to_test)

        # Marcar el mejor modelo
        best_degree_idx = np.argmin(test_errors)
        ax1.axvline(x=degrees_to_test[best_degree_idx], color='red',
                   linestyle='--', alpha=0.5, label=f'Mejor: grado {degrees_to_test[best_degree_idx]}')
        ax1.legend()

        # Gráfico 2: Visualización de múltiples modelos
        X_plot = np.linspace(-3, 3, 300)
        ax2.scatter(X_train, y_train, alpha=0.4, label='Train', s=30)
        ax2.scatter(X_test, y_test, alpha=0.4, label='Test', s=30, marker='s')
        ax2.plot(X, y_true, 'k--', linewidth=2, label='Función real')

        colors = plt.cm.viridis(np.linspace(0, 1, len(degrees_to_test)))
        for i, d in enumerate(degrees_to_test):
            ax2.plot(X_plot, models_predictions[d], linewidth=2,
                    alpha=0.7, color=colors[i], label=f'Grado {d}')

        ax2.set_xlabel('X')
        ax2.set_ylabel('y')
        ax2.set_title('Comparación de Ajustes por Grado')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close()

    st.markdown("""
    **Interpretación:**
    - **Underfitting:** Grado del modelo < complejidad de los datos → alto error en train y test
    - **Overfitting:** Grado del modelo >> complejidad de los datos → bajo error en train, alto en test
    - **Óptimo:** Cuando el error de test es mínimo
    - El modelo ideal tiene grado similar a la complejidad real de los datos
    """)


def render_gradient_descent():
    st.subheader("⚙️ 3. Gradient Descent Interactivo")

    st.markdown("""
    El Gradient Descent es un algoritmo de optimización iterativo para encontrar
    el mínimo de una función. En regresión lineal simple, buscamos minimizar:

    $$J(w_0, w_1) = \\frac{1}{2n} \\sum_{i=1}^{n} (y_i - (w_0 + w_1 x_i))^2$$

    Actualizando los parámetros:
    - $w_0 := w_0 - \\alpha \\frac{\\partial J}{\\partial w_0}$
    - $w_1 := w_1 - \\alpha \\frac{\\partial J}{\\partial w_1}$

    donde $\\alpha$ es el learning rate.
    """)

    # Inicializar estado si no existe
    if 'gd_step' not in st.session_state:
        st.session_state.gd_step = 0
        st.session_state.gd_w0 = 0.0
        st.session_state.gd_w1 = 0.0
        st.session_state.gd_history = [(0.0, 0.0)]
        st.session_state.gd_cost_history = []

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Parámetros")
        # Usar escala logarítmica para el learning rate
        lr_log = st.slider("Learning Rate (α) - escala log", -4.0, -0.3, -1.0, step=0.1, key="gd_lr_log")
        learning_rate = 10 ** lr_log
        st.write(f"**Learning rate actual:** {learning_rate:.4f}")

        n_samples = st.slider("Número de muestras", 10, 100, 30, key="gd_samples")

        if st.button("🔄 Reiniciar", key="gd_reset"):
            st.session_state.gd_step = 0
            st.session_state.gd_w0 = 0.0
            st.session_state.gd_w1 = 0.0
            st.session_state.gd_history = [(0.0, 0.0)]
            st.session_state.gd_cost_history = []
            st.rerun()

    # Generar datos
    np.random.seed(42)
    X = np.linspace(0, 10, n_samples)
    y_true = 2.5 * X + 5
    y = y_true + np.random.normal(0, 2, n_samples)

    # Función de costo
    def compute_cost(w0, w1, X, y):
        m = len(y)
        predictions = w0 + w1 * X
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        return cost

    # Calcular gradientes
    def compute_gradients(w0, w1, X, y):
        m = len(y)
        predictions = w0 + w1 * X
        dw0 = (1/m) * np.sum(predictions - y)
        dw1 = (1/m) * np.sum((predictions - y) * X)
        return dw0, dw1

    # Calcular costo actual
    current_cost = compute_cost(st.session_state.gd_w0, st.session_state.gd_w1, X, y)

    with col2:
        st.markdown("#### Estado actual")
        st.write(f"**Iteración:** {st.session_state.gd_step}")
        st.write(f"**w₀ (intercepto):** {st.session_state.gd_w0:.4f}")
        st.write(f"**w₁ (pendiente):** {st.session_state.gd_w1:.4f}")
        st.write(f"**Costo (MSE):** {current_cost:.4f}")

    # Botones de control
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if st.button("➡️ 1 Paso", key="gd_step1"):
            # Guardar costo actual ANTES de actualizar
            cost = compute_cost(st.session_state.gd_w0, st.session_state.gd_w1, X, y)
            st.session_state.gd_cost_history.append(cost)

            # Calcular gradientes y actualizar pesos
            dw0, dw1 = compute_gradients(st.session_state.gd_w0, st.session_state.gd_w1, X, y)
            st.session_state.gd_w0 -= learning_rate * dw0
            st.session_state.gd_w1 -= learning_rate * dw1
            st.session_state.gd_step += 1
            st.session_state.gd_history.append((st.session_state.gd_w0, st.session_state.gd_w1))
            st.rerun()

    with col_b:
        if st.button("⏩ 10 Pasos", key="gd_step10"):
            for _ in range(10):
                cost = compute_cost(st.session_state.gd_w0, st.session_state.gd_w1, X, y)
                st.session_state.gd_cost_history.append(cost)
                dw0, dw1 = compute_gradients(st.session_state.gd_w0, st.session_state.gd_w1, X, y)
                st.session_state.gd_w0 -= learning_rate * dw0
                st.session_state.gd_w1 -= learning_rate * dw1
                st.session_state.gd_step += 1
                st.session_state.gd_history.append((st.session_state.gd_w0, st.session_state.gd_w1))
            st.rerun()

    with col_c:
        if st.button("⏭️ Converger", key="gd_converge"):
            for _ in range(500):
                cost = compute_cost(st.session_state.gd_w0, st.session_state.gd_w1, X, y)
                st.session_state.gd_cost_history.append(cost)
                dw0, dw1 = compute_gradients(st.session_state.gd_w0, st.session_state.gd_w1, X, y)
                st.session_state.gd_w0 -= learning_rate * dw0
                st.session_state.gd_w1 -= learning_rate * dw1
                st.session_state.gd_step += 1
                st.session_state.gd_history.append((st.session_state.gd_w0, st.session_state.gd_w1))
            st.rerun()

    # Visualizaciones
    fig = plt.figure(figsize=(18, 5))
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Gráfico 1: Datos y línea actual
    y_pred = st.session_state.gd_w0 + st.session_state.gd_w1 * X
    ax1.scatter(X, y, alpha=0.5, label='Datos')
    ax1.plot(X, y_true, 'g--', linewidth=2, label='Relación real')
    ax1.plot(X, y_pred, 'r-', linewidth=2, label='Predicción actual')
    ax1.set_xlabel('X')
    ax1.set_ylabel('y')
    ax1.set_title(f'Regresión - Iteración {st.session_state.gd_step}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gráfico 2: Evolución del costo
    # Si no hay historial de costos pero sí estamos en el paso 0, mostrar el costo inicial
    costs_to_plot = st.session_state.gd_cost_history.copy()
    if len(costs_to_plot) == 0 and st.session_state.gd_step == 0:
        costs_to_plot = [current_cost]

    if len(costs_to_plot) > 0:
        ax2.plot(costs_to_plot, 'b-', linewidth=2, marker='o', markersize=4)
        ax2.set_xlabel('Iteración')
        ax2.set_ylabel('Costo (MSE)')
        ax2.set_title('Evolución del Costo')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Ejecuta algunos pasos\npara ver la evolución',
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_xlabel('Iteración')
        ax2.set_ylabel('Costo (MSE)')
        ax2.set_title('Evolución del Costo')

    # Gráfico 3: Espacio de pesos (w0, w1)
    if len(st.session_state.gd_history) > 1:
        # Crear una malla para el contour plot
        w0_range = np.linspace(-5, 15, 100)
        w1_range = np.linspace(-1, 6, 100)
        W0, W1 = np.meshgrid(w0_range, w1_range)
        Z = np.zeros_like(W0)

        for i in range(W0.shape[0]):
            for j in range(W0.shape[1]):
                Z[i, j] = compute_cost(W0[i, j], W1[i, j], X, y)

        # Dibujar contornos
        contour = ax3.contour(W0, W1, Z, levels=20, cmap='viridis', alpha=0.6)
        ax3.contourf(W0, W1, Z, levels=20, cmap='viridis', alpha=0.3)
        plt.colorbar(contour, ax=ax3, label='Costo')

        # Dibujar la trayectoria del gradient descent
        history = np.array(st.session_state.gd_history)
        ax3.plot(history[:, 0], history[:, 1], 'ro-', linewidth=2, markersize=4, label='Trayectoria')
        ax3.plot(history[0, 0], history[0, 1], 'go', markersize=10, label='Inicio')
        ax3.plot(history[-1, 0], history[-1, 1], 'r*', markersize=15, label='Actual')

        # Marcar el óptimo real (y_true = 2.5 * X + 5, así que w1=2.5, w0=5)
        ax3.plot(5, 2.5, 'k*', markersize=15, label='Óptimo real')

        ax3.set_xlabel('w₀ (intercepto)')
        ax3.set_ylabel('w₁ (pendiente)')
        ax3.set_title('Espacio de Pesos')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Ejecuta algunos pasos\npara ver la trayectoria',
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_xlabel('w₀ (intercepto)')
        ax3.set_ylabel('w₁ (pendiente)')
        ax3.set_title('Espacio de Pesos')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("""
    **Instrucciones:**
    - Usa los botones para avanzar paso a paso y observar cómo converge el algoritmo.
    - Experimenta con diferentes learning rates (escala logarítmica):
      - **α = 0.0001 (-4):** Convergencia muy lenta pero extremadamente estable.
      - **α = 0.01 (-2):** Convergencia lenta pero segura.
      - **α = 0.1 (-1):** Buen balance entre velocidad y estabilidad.
      - **α = 0.5 (-0.3):** Rápido pero puede oscilar.
    - **Gráfico derecho:** Visualiza cómo el algoritmo navega por el espacio de pesos hacia el mínimo.
      Los contornos muestran el valor de la función de costo.
    """)


def render_regularizacion():
    st.subheader("🎯 4. Regularización (L1, L2, Elastic Net)")

    st.markdown("""
    La regularización ayuda a prevenir el overfitting añadiendo una penalización a los coeficientes:

    - **Ridge (L2):** $J = MSE + \\alpha \\sum w_i^2$ (penaliza valores grandes)
    - **Lasso (L1):** $J = MSE + \\alpha \\sum |w_i|$ (puede llevar coeficientes a 0)
    - **Elastic Net:** Combinación de L1 y L2
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Parámetros")
        data_type = st.selectbox("Tipo de datos",
                                ["Sinusoidal", "Lineal", "Cuadrática", "Exponencial", "Paso/Escalón", "Ruido puro"],
                                key="reg_data_type")
        reg_type = st.selectbox("Tipo de regularización",
                                ["Sin regularización", "Ridge (L2)", "Lasso (L1)", "Elastic Net"],
                                key="reg_type")
        # Usar escala logarítmica para alpha
        alpha_log = st.slider("Fuerza de regularización (α) - escala log", -4.0, 2.0, 0.0, step=0.1, key="reg_alpha_log")
        alpha = 10 ** alpha_log
        st.write(f"**α actual:** {alpha:.4f}")

        degree = st.slider("Grado del polinomio", 1, 15, 10, key="reg_degree")
        n_samples = st.slider("Número de muestras", 20, 100, 30, key="reg_samples")
        noise_level = st.slider("Nivel de ruido", 0.0, 0.5, 0.1, key="reg_noise")

    # Generar datos según el tipo seleccionado
    np.random.seed(42)
    X = np.linspace(0, 1, n_samples)

    if data_type == "Sinusoidal":
        y_clean = np.sin(2 * np.pi * X)
    elif data_type == "Lineal":
        y_clean = 2 * X - 0.5
    elif data_type == "Cuadrática":
        y_clean = 4 * (X - 0.5) ** 2 - 0.5
    elif data_type == "Exponencial":
        y_clean = np.exp(2 * X) / np.exp(2) - 0.5
    elif data_type == "Paso/Escalón":
        y_clean = np.where(X < 0.5, -0.5, 0.5)
    else:  # Ruido puro
        y_clean = np.zeros_like(X)

    y = y_clean + np.random.normal(0, noise_level, n_samples)

    # Crear características polinómicas
    X_reshaped = X.reshape(-1, 1)
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X_reshaped)

    # Entrenar modelo según el tipo
    if reg_type == "Sin regularización":
        model = LinearRegression()
    elif reg_type == "Ridge (L2)":
        model = Ridge(alpha=alpha)
    elif reg_type == "Lasso (L1)":
        model = Lasso(alpha=alpha, max_iter=10000)
    else:  # Elastic Net
        model = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)

    model.fit(X_poly, y)

    # Predicciones
    X_plot = np.linspace(0, 1, 300)

    # Calcular y_clean para X_plot (función real sin ruido)
    if data_type == "Sinusoidal":
        y_plot_clean = np.sin(2 * np.pi * X_plot)
    elif data_type == "Lineal":
        y_plot_clean = 2 * X_plot - 0.5
    elif data_type == "Cuadrática":
        y_plot_clean = 4 * (X_plot - 0.5) ** 2 - 0.5
    elif data_type == "Exponencial":
        y_plot_clean = np.exp(2 * X_plot) / np.exp(2) - 0.5
    elif data_type == "Paso/Escalón":
        y_plot_clean = np.where(X_plot < 0.5, -0.5, 0.5)
    else:  # Ruido puro
        y_plot_clean = np.zeros_like(X_plot)

    X_plot_reshaped = X_plot.reshape(-1, 1)
    X_plot_poly = poly_features.transform(X_plot_reshaped)
    y_plot = model.predict(X_plot_poly)
    y_pred = model.predict(X_poly)

    # Métricas
    mse = np.mean((y - y_pred) ** 2)

    with col2:
        st.markdown("#### Resultados")
        st.write(f"**Tipo de datos:** {data_type}")
        st.write(f"**Regularización:** {reg_type}")
        st.write(f"**MSE:** {mse:.4f}")
        st.write(f"**Coeficientes no nulos:** {np.sum(np.abs(model.coef_) > 0.001)}/{len(model.coef_)}")
        st.write(f"**Norma L2 de coeficientes:** {np.linalg.norm(model.coef_):.4f}")

    # Visualizaciones
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico 1: Ajuste
    ax1.scatter(X, y, alpha=0.5, label='Datos con ruido', s=30)
    ax1.plot(X_plot, y_plot_clean, 'g--', linewidth=2, label='Función real')
    ax1.plot(X_plot, y_plot, 'r-', linewidth=2, label='Modelo')
    ax1.set_xlabel('X')
    ax1.set_ylabel('y')
    ax1.set_title(f'{data_type} - {reg_type}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Ajustar límites dinámicamente
    y_min = min(y.min(), y_plot.min(), y_plot_clean.min()) - 0.3
    y_max = max(y.max(), y_plot.max(), y_plot_clean.max()) + 0.3
    ax1.set_ylim([y_min, y_max])

    # Gráfico 2: Magnitud de coeficientes
    coef_indices = np.arange(len(model.coef_))
    ax2.bar(coef_indices, np.abs(model.coef_))
    ax2.set_xlabel('Índice del coeficiente')
    ax2.set_ylabel('Magnitud |w_i|')
    ax2.set_title('Magnitud de los coeficientes')
    ax2.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close()

    st.markdown("""
    **Observaciones:**
    - **Sin regularización:** Con grados altos, el modelo se ajusta demasiado al ruido (overfitting).
    - **Ridge (L2):** Reduce la magnitud de todos los coeficientes uniformemente, suaviza la curva.
    - **Lasso (L1):** Puede llevar algunos coeficientes exactamente a cero (selección de características).
    - **Elastic Net:** Combina las ventajas de Ridge y Lasso.

    **Experimenta con:**
    - **Datos lineales:** La regularización tiene poco efecto con grados bajos (1-2).
    - **Datos sinusoidales/cuadráticos:** Necesitan grados medios (3-5), la regularización ayuda a evitar overfitting.
    - **Paso/Escalón:** Funciones discontinuas son difíciles de ajustar, la regularización ayuda a suavizar.
    - **Ruido puro:** La regularización es esencial para evitar ajustar el ruido aleatorio.

    Aumenta α para reducir overfitting (pero aumenta el bias). Ajusta el nivel de ruido para ver el efecto.
    """)


def render_regresion_logistica():
    st.subheader("🔵🔴 5. Regresión Logística")

    st.markdown("""
    La regresión logística se usa para clasificación binaria. Modela la probabilidad de que
    una muestra pertenezca a una clase usando la función sigmoide:

    $$P(y=1|x) = \\frac{1}{1 + e^{-(w_0 + w_1 x_1 + w_2 x_2)}}$$
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Parámetros")
        n_samples = st.slider("Número de muestras", 50, 300, 150, key="logr_samples")
        separation = st.slider("Separación de clases", 0.5, 3.0, 1.5, key="logr_sep")
        class_weight = st.selectbox("Balance de clases",
                                    ["Balanceado", "Desbalanceado (70-30)"],
                                    key="logr_weight")

    # Generar datos
    weights = [0.5, 0.5] if class_weight == "Balanceado" else [0.7, 0.3]
    np.random.seed(42)
    X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0,
                              n_informative=2, n_clusters_per_class=1,
                              class_sep=separation, weights=weights, random_state=42)

    # Entrenar modelo
    model = LogisticRegression()
    model.fit(X, y)

    # Métricas
    accuracy = model.score(X, y)

    with col2:
        st.markdown("#### Resultados")
        st.write(f"**Accuracy:** {accuracy:.3f}")
        st.write(f"**Intercepto:** {model.intercept_[0]:.3f}")
        st.write(f"**Coeficientes:** [{model.coef_[0][0]:.3f}, {model.coef_[0][1]:.3f}]")

        # Contar clases
        unique, counts = np.unique(y, return_counts=True)
        st.write(f"**Clase 0:** {counts[0]} muestras")
        st.write(f"**Clase 1:** {counts[1]} muestras")

    # Crear malla para visualizar probabilidades
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Predecir probabilidades
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    # Visualización
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico 1: Mapa de probabilidades
    contour = ax1.contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.8)
    ax1.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    ax1.scatter(X[y==0, 0], X[y==0, 1], c='blue', edgecolors='k', marker='o',
               s=50, alpha=0.7, label='Clase 0')
    ax1.scatter(X[y==1, 0], X[y==1, 1], c='red', edgecolors='k', marker='s',
               s=50, alpha=0.7, label='Clase 1')
    ax1.set_xlabel('X₁')
    ax1.set_ylabel('X₂')
    ax1.set_title('Probabilidades de la Clase 1')
    ax1.legend()
    plt.colorbar(contour, ax=ax1, label='P(y=1|x)')

    # Gráfico 2: Frontera de decisión con más detalle
    ax2.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['lightblue', 'lightcoral'], alpha=0.5)
    ax2.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=3, linestyles='--')
    ax2.scatter(X[y==0, 0], X[y==0, 1], c='blue', edgecolors='k', marker='o',
               s=50, alpha=0.7, label='Clase 0')
    ax2.scatter(X[y==1, 0], X[y==1, 1], c='red', edgecolors='k', marker='s',
               s=50, alpha=0.7, label='Clase 1')
    ax2.set_xlabel('X₁')
    ax2.set_ylabel('X₂')
    ax2.set_title('Frontera de Decisión (P=0.5)')
    ax2.legend()

    st.pyplot(fig)
    plt.close()

    st.markdown("""
    **Interpretación:**
    - **Izquierda:** Mapa de calor mostrando las probabilidades. Rojo = alta probabilidad de Clase 1,
      Azul = alta probabilidad de Clase 0.
    - **Derecha:** Frontera de decisión (línea negra discontinua) donde P(y=1) = 0.5.
    - Los círculos azules son muestras de Clase 0, los cuadrados rojos son Clase 1.
    - La regresión logística crea una frontera de decisión lineal.
    """)


def render_red_neuronal():
    st.subheader("🧠 6. Red Neuronal Sencilla")

    st.markdown("""
    Una red neuronal simple con una capa oculta puede verse como una extensión de la regresión logística.
    Vamos a implementar una red para clasificación binaria con activación sigmoide.

    **Arquitectura:** Input (2) → Hidden Layer (n neuronas) → Output (1)
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Parámetros")
        n_hidden = st.slider("Neuronas en capa oculta", 2, 20, 4, key="nn_hidden")
        learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.5, key="nn_lr")
        n_samples = st.slider("Número de muestras", 50, 300, 100, key="nn_samples")
        pattern = st.selectbox("Patrón de datos",
                              ["Linealmente separable", "XOR (no lineal)", "Círculos"],
                              key="nn_pattern")

    # Generar datos según el patrón
    np.random.seed(42)
    if pattern == "Linealmente separable":
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0,
                                  n_informative=2, n_clusters_per_class=1,
                                  class_sep=2.0, random_state=42)
    elif pattern == "XOR (no lineal)":
        X = np.random.randn(n_samples, 2)
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
    else:  # Círculos
        radius = np.random.rand(n_samples)
        angle = 2 * np.pi * np.random.rand(n_samples)
        X = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
        y = (radius > 0.5).astype(int)

    # Implementación simple de red neuronal
    class SimpleNeuralNetwork:
        def __init__(self, n_hidden):
            self.n_hidden = n_hidden
            # Inicialización aleatoria de pesos
            self.W1 = np.random.randn(2, n_hidden) * 0.5
            self.b1 = np.zeros((1, n_hidden))
            self.W2 = np.random.randn(n_hidden, 1) * 0.5
            self.b2 = np.zeros((1, 1))

        def sigmoid(self, z):
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

        def forward(self, X):
            self.z1 = np.dot(X, self.W1) + self.b1
            self.a1 = self.sigmoid(self.z1)
            self.z2 = np.dot(self.a1, self.W2) + self.b2
            self.a2 = self.sigmoid(self.z2)
            return self.a2

        def train_step(self, X, y, lr):
            m = X.shape[0]

            # Forward pass
            self.forward(X)

            # Backward pass
            dz2 = self.a2 - y.reshape(-1, 1)
            dW2 = (1/m) * np.dot(self.a1.T, dz2)
            db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)

            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * self.a1 * (1 - self.a1)
            dW1 = (1/m) * np.dot(X.T, dz1)
            db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)

            # Update weights
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2

        def predict(self, X):
            return (self.forward(X) > 0.5).astype(int)

    # Entrenar red neuronal
    nn = SimpleNeuralNetwork(n_hidden)
    losses = []

    for epoch in range(1000):
        nn.train_step(X, y, learning_rate)
        if epoch % 50 == 0:
            predictions = nn.forward(X)
            loss = -np.mean(y.reshape(-1, 1) * np.log(predictions + 1e-8) +
                          (1 - y.reshape(-1, 1)) * np.log(1 - predictions + 1e-8))
            losses.append(loss)

    # Métricas
    y_pred = nn.predict(X)
    accuracy = np.mean(y_pred.flatten() == y)

    with col2:
        st.markdown("#### Resultados")
        st.write(f"**Neuronas ocultas:** {n_hidden}")
        st.write(f"**Accuracy:** {accuracy:.3f}")
        st.write(f"**Pérdida final:** {losses[-1]:.4f}")
        st.write(f"**Épocas de entrenamiento:** 1000")

    # Visualizaciones
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico 1: Frontera de decisión
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    Z = nn.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    contour = ax1.contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.8)
    ax1.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    ax1.scatter(X[y==0, 0], X[y==0, 1], c='blue', edgecolors='k', marker='o',
               s=50, alpha=0.7, label='Clase 0')
    ax1.scatter(X[y==1, 0], X[y==1, 1], c='red', edgecolors='k', marker='s',
               s=50, alpha=0.7, label='Clase 1')
    ax1.set_xlabel('X₁')
    ax1.set_ylabel('X₂')
    ax1.set_title(f'Frontera de Decisión ({n_hidden} neuronas ocultas)')
    ax1.legend()
    plt.colorbar(contour, ax=ax1, label='Probabilidad')

    # Gráfico 2: Evolución de la pérdida
    ax2.plot(range(0, 1000, 50), losses, 'b-', linewidth=2)
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Pérdida (Cross-Entropy)')
    ax2.set_title('Evolución del Entrenamiento')
    ax2.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close()

    st.markdown("""
    **Ventajas de las redes neuronales:**
    - Pueden aprender patrones no lineales (prueba con "XOR" o "Círculos").
    - Con más neuronas ocultas, pueden modelar relaciones más complejas.
    - La capa oculta aprende representaciones útiles de los datos.

    **Observaciones:**
    - Con patrones linealmente separables, la red se comporta como regresión logística.
    - Con patrones no lineales (XOR, círculos), la red puede crear fronteras de decisión complejas.
    - Más neuronas ocultas = más capacidad, pero también más riesgo de overfitting.
    """)
