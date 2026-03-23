"""
Sistema de Predicción de Tarifas Eléctricas DB1
Prototipo Interactivo - Modelo SARIMA

Proyecto: Seminario de Innovación en Análisis y Visualización de Datos
Institución: Universidad Internacional de La Rioja (UNIR)
Autores: Abraham López Velázquez, Fernando Abraham Deras Stenner
Año: 2025

Descripción:
Este prototipo permite generar predicciones de tarifas eléctricas DB1
utilizando el modelo SARIMA(0,1,1)(0,1,1,12) entrenado. El usuario puede
seleccionar el horizonte de predicción y el sistema ejecuta el modelo
en tiempo real para generar las proyecciones correspondientes.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN DE LA APLICACIÓN
# ============================================================================

st.set_page_config(
    page_title="Sistema de Predicción - Tarifas DB1",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    /* Tipografía principal */
    .main {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Título principal */
    h1 {
        color: #1f4788;
        font-weight: 600;
        border-bottom: 3px solid #1f4788;
        padding-bottom: 10px;
    }
    
    /* Subtítulos */
    h2 {
        color: #2c5aa0;
        font-weight: 500;
        margin-top: 30px;
    }
    
    h3 {
        color: #34495e;
        font-weight: 500;
    }
    
    /* Botón principal */
    .stButton>button {
        background-color: #1f4788;
        color: white;
        font-size: 16px;
        font-weight: 500;
        padding: 12px 24px;
        border-radius: 4px;
        border: none;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #163761;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        font-size: 24px;
        color: #1f4788;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 4px;
    }
    
    /* Tablas */
    .dataframe {
        font-size: 14px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #7f8c8d;
        padding: 20px 0;
        margin-top: 40px;
        border-top: 1px solid #ecf0f1;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES DE CARGA DE DATOS Y MODELO
# ============================================================================

@st.cache_resource
def cargar_modelo():
    """
    Carga el modelo SARIMA entrenado desde archivo pickle.
    
    Returns:
        Modelo SARIMA entrenado
    
    Raises:
        FileNotFoundError: Si el archivo del modelo no existe
    """
    try:
        with open('modelos/modelo_sarima_db1.pkl', 'rb') as f:
            modelo = pickle.load(f)
        return modelo
    except FileNotFoundError:
        st.error("Error: Modelo no encontrado en 'modelos/modelo_sarima_db1.pkl'")
        st.info("Ejecute primero el Jupyter Notebook con las celdas de exportación para generar el modelo.")
        st.stop()
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        st.stop()

@st.cache_data
def cargar_datos_historicos():
    """
    Carga la serie temporal histórica de tarifas DB1.
    
    Returns:
        DataFrame con datos históricos indexados por fecha
    
    Raises:
        FileNotFoundError: Si el archivo de datos no existe
    """
    try:
        df = pd.read_csv('datos/datos_tarifas_db1.csv', index_col=0, parse_dates=True)
        return df
    except FileNotFoundError:
        st.error("Error: Datos históricos no encontrados en 'datos/datos_tarifas_db1.csv'")
        st.info("Ejecute primero el Jupyter Notebook con las celdas de exportación para generar los datos.")
        st.stop()
    except Exception as e:
        st.error(f"Error al cargar datos históricos: {str(e)}")
        st.stop()

@st.cache_data
def cargar_metricas():
    """
    Carga las métricas de rendimiento del modelo.
    
    Returns:
        Diccionario con métricas (MAE, RMSE, MAPE, R2, AIC, BIC)
    """
    try:
        with open('datos/metricas_modelo.json', 'r') as f:
            metricas = json.load(f)
        return metricas
    except FileNotFoundError:
        return None
    except Exception as e:
        st.warning(f"Advertencia: No se pudieron cargar las métricas del modelo")
        return None

# ============================================================================
# FUNCIÓN PRINCIPAL: GENERACIÓN DE PREDICCIONES
# ============================================================================

def generar_prediccion(modelo, datos_historicos, num_meses):
    """
    Ejecuta el modelo SARIMA para generar predicciones futuras.
    
    Esta función ejecuta el modelo en tiempo real, calculando predicciones
    nuevas basadas en el horizonte temporal especificado por el usuario.
    
    Args:
        modelo: Modelo SARIMA entrenado
        datos_historicos: Serie temporal histórica
        num_meses: Número de meses a predecir
    
    Returns:
        tuple: (DataFrame con predicciones, valores predichos, 
                intervalo de confianza, fechas futuras)
    """
    # Generar forecast utilizando el modelo entrenado
    forecast = modelo.get_forecast(steps=num_meses)
    predicciones = forecast.predicted_mean
    intervalo_confianza = forecast.conf_int()
    
    # Generar fechas futuras
    ultima_fecha = datos_historicos.index[-1]
    fechas_futuras = pd.date_range(
        start=ultima_fecha + pd.DateOffset(months=1),
        periods=num_meses,
        freq='MS'
    )
    
    # Crear DataFrame estructurado con resultados
    df_resultados = pd.DataFrame({
        'Fecha': fechas_futuras,
        'Prediccion': predicciones.values,
        'Limite_Inferior_IC95': intervalo_confianza.iloc[:, 0].values,
        'Limite_Superior_IC95': intervalo_confianza.iloc[:, 1].values
    })
    
    return df_resultados, predicciones, intervalo_confianza, fechas_futuras

# ============================================================================
# INICIALIZACIÓN: CARGA DE DATOS Y MODELO
# ============================================================================

modelo = cargar_modelo()
datos_historicos = cargar_datos_historicos()
metricas = cargar_metricas()

# ============================================================================
# INTERFAZ DE USUARIO
# ============================================================================

# Encabezado principal
st.title("Sistema de Predicción de Tarifas Eléctricas DB1")
st.markdown("**Prototipo Interactivo | Modelo SARIMA en Tiempo Real**")

st.markdown("---")

# Sección de información
with st.expander("Información del Sistema", expanded=False):
    st.markdown("""
    ### Descripción
    
    Este sistema permite generar predicciones de tarifas eléctricas DB1 (Doméstico de Bajo Consumo)
    utilizando un modelo SARIMA entrenado con datos históricos del periodo 2017-2025.
    
    ### Funcionamiento
    
    1. **Configuración:** Seleccione el horizonte de predicción (número de meses)
    2. **Ejecución:** El modelo SARIMA se ejecuta en tiempo real al presionar el botón
    3. **Resultados:** Se generan predicciones con intervalos de confianza al 95%
    4. **Análisis:** Visualización gráfica y tabular de las proyecciones
    
    ### Especificaciones Técnicas
    
    - **Modelo:** SARIMA(0,1,1)(0,1,1,12)
    - **Periodicidad:** Mensual con estacionalidad anual
    - **Horizonte:** 1 a 24 meses
    - **Intervalo de Confianza:** 95%
    """)

st.markdown("---")

# Información del modelo
st.subheader("Información del Modelo")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Modelo",
        value="SARIMA(0,1,1)(0,1,1,12)"
    )

with col2:
    num_observaciones = len(datos_historicos)
    st.metric(
        label="Observaciones Históricas",
        value=f"{num_observaciones}"
    )

with col3:
    precio_actual = datos_historicos.iloc[-1, 0]
    st.metric(
        label="Precio Actual (MXN/kWh)",
        value=f"${precio_actual:.4f}"
    )

with col4:
    if metricas:
        mape = metricas.get('MAPE', 0)
        st.metric(
            label="Precisión (MAPE)",
            value=f"{mape:.2f}%"
        )
    else:
        st.metric(
            label="Precisión (MAPE)",
            value="N/D"
        )

st.markdown("---")

# Panel de control
st.subheader("Configuración de Predicción")

col_control1, col_control2 = st.columns([3, 1])

with col_control1:
    meses_a_predecir = st.slider(
        label="Seleccione el horizonte de predicción (meses):",
        min_value=1,
        max_value=24,
        value=12,
        help="Número de meses futuros para los cuales se generarán las predicciones"
    )
    
    st.caption(f"Se generarán predicciones para los próximos {meses_a_predecir} meses")

with col_control2:
    st.write("")
    st.write("")
    ejecutar_prediccion = st.button(
        "Generar Predicción",
        type="primary",
        use_container_width=True
    )

st.markdown("---")

# ============================================================================
# PROCESAMIENTO Y VISUALIZACIÓN DE RESULTADOS
# ============================================================================

if ejecutar_prediccion:
    
    # Indicador de procesamiento
    with st.spinner(f'Ejecutando modelo SARIMA para {meses_a_predecir} meses...'):
        
        # Ejecutar modelo
        df_pred, vals_pred, ic, fechas = generar_prediccion(
            modelo,
            datos_historicos,
            meses_a_predecir
        )
    
    st.success(f"Predicción generada exitosamente para {meses_a_predecir} meses")
    
    # ========================================================================
    # Sección 1: Visualización Gráfica
    # ========================================================================
    
    st.subheader("Resultados: Visualización Gráfica")
    
    # Crear figura
    fig = go.Figure()
    
    # Datos históricos (últimos 36 meses para contexto)
    n_historico = min(36, len(datos_historicos))
    historico_viz = datos_historicos.tail(n_historico)
    
    fig.add_trace(go.Scatter(
        x=historico_viz.index,
        y=historico_viz.iloc[:, 0],
        mode='lines',
        name='Datos Históricos',
        line=dict(color='#34495e', width=2),
        hovertemplate='<b>%{x|%Y-%m}</b><br>Precio: $%{y:.4f}<extra></extra>'
    ))
    
    # Predicciones
    fig.add_trace(go.Scatter(
        x=fechas,
        y=vals_pred,
        mode='lines+markers',
        name='Predicción',
        line=dict(color='#e74c3c', width=2.5, dash='dash'),
        marker=dict(size=6, symbol='circle'),
        hovertemplate='<b>%{x|%Y-%m}</b><br>Predicción: $%{y:.4f}<extra></extra>'
    ))
    
    # Intervalo de confianza 95%
    fig.add_trace(go.Scatter(
        x=list(fechas) + list(fechas[::-1]),
        y=list(ic.iloc[:, 1]) + list(ic.iloc[::-1, 0]),
        fill='toself',
        fillcolor='rgba(231, 76, 60, 0.15)',
        line=dict(color='rgba(231, 76, 60, 0)'),
        name='Intervalo de Confianza 95%',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Configuración del layout
    fig.update_layout(
        title={
            'text': f'Predicción de Tarifas DB1 - Horizonte: {meses_a_predecir} Meses',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#2c3e50'}
        },
        xaxis=dict(
            title='Periodo',
            showgrid=True,
            gridcolor='#ecf0f1'
        ),
        yaxis=dict(
            title='Precio (MXN/kWh)',
            showgrid=True,
            gridcolor='#ecf0f1'
        ),
        hovermode='x unified',
        height=550,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Segoe UI, sans-serif', size=12, color='#2c3e50'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # Sección 2: Tabla de Resultados
    # ========================================================================
    
    st.subheader("Resultados: Tabla de Predicciones")
    
    # Formatear datos para visualización
    df_display = df_pred.copy()
    df_display['Fecha'] = df_display['Fecha'].dt.strftime('%Y-%m')
    df_display.columns = ['Fecha', 'Predicción ($/kWh)', 'Límite Inferior (IC 95%)', 'Límite Superior (IC 95%)']
    
    # Formatear valores numéricos
    for col in df_display.columns[1:]:
        df_display[col] = df_display[col].apply(lambda x: f"${x:.4f}")
    
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True
    )
    
    # ========================================================================
    # Sección 3: Análisis Estadístico
    # ========================================================================
    
    st.subheader("Análisis Estadístico")
    
    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
    
    with col_a1:
        precio_inicial_pred = vals_pred.iloc[0]
        st.metric(
            label="Precio Inicial Proyectado",
            value=f"${precio_inicial_pred:.4f}"
        )
    
    with col_a2:
        precio_final_pred = vals_pred.iloc[-1]
        st.metric(
            label="Precio Final Proyectado",
            value=f"${precio_final_pred:.4f}"
        )
    
    with col_a3:
        cambio_absoluto = precio_final_pred - precio_actual
        st.metric(
            label="Cambio Absoluto",
            value=f"${cambio_absoluto:.4f}",
            delta=f"${cambio_absoluto:.4f}"
        )
    
    with col_a4:
        cambio_porcentual = ((precio_final_pred - precio_actual) / precio_actual) * 100
        st.metric(
            label="Cambio Porcentual",
            value=f"{cambio_porcentual:+.2f}%",
            delta=f"{cambio_porcentual:+.2f}%"
        )
    
    # ========================================================================
    # Sección 4: Exportación de Resultados
    # ========================================================================
    
    st.markdown("---")
    st.subheader("Exportación de Resultados")
    
    # Preparar CSV
    csv_export = df_pred.to_csv(index=False, float_format='%.6f')
    
    col_exp1, col_exp2 = st.columns([2, 1])
    
    with col_exp1:
        st.download_button(
            label="Descargar Predicciones (CSV)",
            data=csv_export,
            file_name=f"prediccion_tarifas_db1_{meses_a_predecir}meses_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_exp2:
        st.caption("Formato: CSV con precisión de 6 decimales")

else:
    # Mensaje cuando no se ha ejecutado predicción
    st.info("Configure el horizonte de predicción y presione 'Generar Predicción' para ejecutar el modelo SARIMA")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div class='footer'>
    <p><strong>Sistema de Predicción de Tarifas Eléctricas DB1</strong></p>
    <p>Seminario de Innovación en Análisis y Visualización de Datos | UNIR 2025</p>
    <p>Autores: Abraham López Velázquez, Fernando Abraham Deras Stenner</p>
    <p style='font-size: 11px; margin-top: 15px; color: #95a5a6;'>
        Modelo: SARIMA(0,1,1)(0,1,1,12) | Datos: Diciembre 2017 - Septiembre 2025 | Precisión: MAPE < 1%
    </p>
</div>
""", unsafe_allow_html=True)