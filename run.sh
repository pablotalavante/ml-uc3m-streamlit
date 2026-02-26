#!/bin/bash

# Script de inicio rápido para la aplicación de Machine Learning

echo "🚀 Iniciando aplicación de Machine Learning..."
echo ""

# Verificar si streamlit está instalado
if ! command -v streamlit &> /dev/null
then
    echo "❌ Streamlit no está instalado."
    echo "📦 Instalando dependencias..."
    pip install -r requirements.txt
    echo ""
fi

echo "✅ Iniciando Streamlit..."
echo "📝 La aplicación se abrirá en http://localhost:8501"
echo ""

streamlit run app.py
