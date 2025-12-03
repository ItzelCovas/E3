import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# librerías instaladas: pip install streamlit sentence-transformers pandas numpy plotly scikit-learn

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Analizador de Taylor Swift", layout="wide")
st.title("Comparador Semántico: Edición Taylor Swift")

# --- PASO 1: CARGAR DATOS REALES ---
@st.cache_data
def load_data():
    # Cargar el archivo CSV
    # Asegúrate de que 'TaylorSwift.csv' esté en la misma carpeta que este script
    df = pd.read_csv('TaylorSwift.csv')
    
    # LIMPIEZA DE DATOS:
    # 1. Eliminar filas donde no haya letra (Lyric es NaN)
    df = df.dropna(subset=['Lyric'])
    # 2. Resetear el índice para evitar problemas futuros
    df = df.reset_index(drop=True)
    return df

try:
    df = load_data()
    st.write(f"✅ Se cargaron correctamente {len(df)} canciones de Taylor Swift.")
except FileNotFoundError:
    st.error("❌ No encuentro el archivo 'TaylorSwift.csv'. Asegúrate de ponerlo en la misma carpeta.")
    st.stop()

# --- PASO 2: CARGAR MODELO DE IA ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

st.write("---")
st.header("Nivel 1: Comparación Global")

# --- PASO 3: CALCULAR EMBEDDINGS ---
if 'embeddings' not in st.session_state:
    with st.spinner('Procesando letras de Taylor Swift... (esto tardará un poco la primera vez)'):
        # Usamos la columna real 'Lyric'
        st.session_state.embeddings = model.encode(df['Lyric'].tolist())
    st.success("¡Análisis semántico completado!")

# --- PASO 4: SELECCIÓN DE CANCIÓN BASE ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Selecciona una canción (Texto 1)")
    # Usamos la columna real 'Title'
    opciones_titulos = df['Title'].tolist()
    seleccion_titulo = st.selectbox("Elige una canción:", opciones_titulos)
    
    # Encontrar índice
    idx_seleccion = df[df['Title'] == seleccion_titulo].index[0]
    texto_1 = df.iloc[idx_seleccion]['Lyric']
    
    st.text_area("Letra:", value=texto_1, height=300)

# --- PASO 5: ENCONTRAR SIMILARES ---
with col2:
    st.subheader("Canciones más similares semánticamente")
    n_similares = st.slider("Número de coincidencias:", 1, 5, 3)
    
    if st.button("🔍 Buscar Similares"):
        vector_seleccionado = st.session_state.embeddings[idx_seleccion].reshape(1, -1)
        similitudes = cosine_similarity(vector_seleccionado, st.session_state.embeddings)[0]
        
        # Crear DF temporal
        df_resultados = df.copy()
        df_resultados['Similitud'] = similitudes
        
        # Ordenar y filtrar la misma canción
        df_resultados = df_resultados.sort_values(by='Similitud', ascending=False)
        df_resultados = df_resultados[df_resultados['Title'] != seleccion_titulo]
        
        top_n = df_resultados.head(n_similares)
        
        for i, row in top_n.iterrows():
            with st.expander(f"Top {i+1}: {row['Title']} (Similitud: {row['Similitud']:.2%})"):
                st.write(f"**Álbum:** {row['Album']}")
                st.progress(float(row['Similitud']))
                st.caption(row['Lyric'][:200] + "...")
