import streamlit as st
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Comparador Taylor Swift", layout="wide")
st.title("Comparador de letras de Taylor Swift")

# carga de datos
@st.cache_data
def load_data():
    try:
        raw_df = pd.read_csv('taylor_swift_lyrics.csv', encoding='latin1')
        df_grouped = raw_df.groupby(['track_title', 'album', 'year'])['lyric'].apply(lambda x: '\n'.join(x)).reset_index()
        return df_grouped
    except FileNotFoundError:
        return None

df = load_data()
if df is None:
    st.error("Falta el archivo 'taylor_swift_lyrics.csv'.")
    st.stop()

# carga del modelo
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# embedding
if 'embeddings' not in st.session_state:
    with st.spinner('Procesando letras...'):
        st.session_state.embeddings = model.encode(df['lyric'].tolist())

def split_text(text):
    return [line for line in text.split('\n') if line.strip()]

#Interfaz  
# dos pestañas principales
tab1, tab2 = st.tabs(["Análisis Detallado (Nivel 1 y 2)", "Matriz Global (Colección)"])

# PESTAÑA 1
with tab1:
    c1, c2 = st.columns([2, 1])
    with c1:
        titulo_1 = st.selectbox("1. Selecciona la canción base:", df['track_title'].unique())
    with c2:
        n_similares = st.slider("2. ¿Cuántas similares buscar?", 1, 10, 5)

    # Lógica, no visual
    idx_1 = df[df['track_title'] == titulo_1].index[0]
    texto_1 = df.iloc[idx_1]['lyric']
    vec_1 = st.session_state.embeddings[idx_1].reshape(1, -1)
    sims = cosine_similarity(vec_1, st.session_state.embeddings)[0]
    df['Similitud'] = sims
    df_sorted = df.sort_values('Similitud', ascending=False)
    df_sorted = df_sorted[df_sorted['track_title'] != titulo_1] # Excluir la misma

    st.divider()

    # lista de resultados (nivel 1)
    st.subheader(f"Top {n_similares} canciones más parecidas a '{titulo_1}':")
    top_n = df_sorted.head(n_similares)

    # tabla
    st.dataframe(
        top_n[['track_title', 'album', 'Similitud']].style.format({'Similitud': '{:.2%}'}),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    #SELECCIÓN PARA COMPARAR (NIVEL 2)
    st.subheader("Comparación Detallada")
    col_sel, col_btn = st.columns([3, 1])

    with col_sel:
        opciones = top_n['track_title'].tolist()
        titulo_2 = st.selectbox("Elige una de la lista de arriba para comparar:", opciones)
        
        # datos de la segunda canción
        idx_2 = df[df['track_title'] == titulo_2].index[0]
        texto_2 = df.iloc[idx_2]['lyric']

    # vista de las letras
    col_txt1, col_txt2 = st.columns(2)

    with col_txt1:
        st.markdown(f"**- {titulo_1}** (Base)")
        st.text_area("Letra 1", texto_1, height=300, label_visibility="collapsed")

    with col_txt2:
        st.markdown(f"**- {titulo_2}** (Comparación)")
        st.text_area("Letra 2", texto_2, height=300, label_visibility="collapsed")

    # 5. MAPA DE CALOR
    st.markdown("<br>", unsafe_allow_html=True)    
    if st.button("Generar Mapa de Calor (Frase a Frase)", use_container_width=True, type="primary"):
        
        frases_1 = split_text(texto_1)
        frases_2 = split_text(texto_2)
        
        if len(frases_1) > 0 and len(frases_2) > 0:
            emb_1 = model.encode(frases_1)
            emb_2 = model.encode(frases_2)
            matriz = cosine_similarity(emb_1, emb_2)
            
            c1, c_mapa, c2 = st.columns([1, 4, 1]) 
            
            with c_mapa:
                fig = px.imshow(
                    matriz,
                    x=[f"T2: {f[:25]}..." for f in frases_2],
                    y=[f"T1: {f[:25]}..." for f in frases_1],
                    color_continuous_scale="RdBu_r",
                    aspect="equal",
                    title=f"Similitud: {titulo_1} vs. {titulo_2}"
                )
                
                fig.update_layout(
                    width=1000,
                    height=1000,
                    xaxis_tickangle=-90, 
                    autosize=False,
                    
                    title_x=0.53,         
                    title_xanchor='center',
                    title_y=0.95,

                    #margin=dict(t=100, b=90, l=30, r=50),  
                    
                    coloraxis_colorbar=dict(
                        len=0.7,
                        yanchor="middle",
                        y=0.5,
                        title=dict(text="Similitud", side="right")
                    )
                )

                st.plotly_chart(fig, use_container_width=False)
                
        else:
            st.warning("Letras insuficientes para comparar.")

# PESTAÑA 2
with tab2:
    st.header("Matriz de Similitud Global entre Todas las Canciones")
    
    st.markdown("""
    Esta matriz cruza **todas las canciones contra todas**.
    
    **Guía de colores:**
    * 🟨 **Amarillo brillante:** Alta similitud (canciones muy parecidas).
    * 🟦 **Morado oscuro:** Baja similitud (temáticas opuestas).
    * 📉 **La Diagonal:** Siempre es amarilla porque compara cada canción consigo misma.
    """)
    
    st.write("Se muestra cómo se relacionan todas las canciones entre sí.")
    
    if st.button("Calcular Matriz total"):
        with st.spinner("Calculando interacciones entre todas las canciones..."):
            matriz_global = cosine_similarity(st.session_state.embeddings)
            
            fig_g = px.imshow(
                matriz_global,
                x=df['track_title'],
                y=df['track_title'],
                color_continuous_scale="Viridis",
                title="Matriz de Similitud Global"
            )
            fig_g.update_layout(width=900, height=900)
            st.plotly_chart(fig_g, use_container_width=True)