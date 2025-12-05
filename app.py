import streamlit as st
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configuracion basica de la pagina
st.set_page_config(page_title="Comparador Taylor Swift", layout="wide")
st.title("Comparador de letras de Taylor Swift")

# Carga de datos
# Utiliza la cache para optimizar la velocidad y evitar recargas innecesarias
@st.cache_data
def load_data():
    try:
        raw_df = pd.read_csv('taylor_swift_lyrics.csv', encoding='latin1')
        # Agrupa las lineas del archivo CSV para reconstruir la cancion completa
        df_grouped = raw_df.groupby(['track_title', 'album', 'year'])['lyric'].apply(lambda x: '\n'.join(x)).reset_index()
        return df_grouped
    except FileNotFoundError:
        return None

df = load_data()
if df is None:
    st.error("Falta el archivo 'taylor_swift_lyrics.csv'.")
    st.stop()

# Carga del modelo de IA
# Carga el modelo que convierte texto en vectores numericos
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Generacion de Embeddings
# Almacena los vectores en memoria para calcularlos solo una vez al inicio
if 'embeddings' not in st.session_state:
    with st.spinner('Procesando letras...'):
        st.session_state.embeddings = model.encode(df['lyric'].tolist())

# Funcion auxiliar que divide el texto en frases individuales
def split_text(text):
    return [line for line in text.split('\n') if line.strip()]

# Interfaz grafica
# Genera dos pestañas para organizar las vistas de la herramienta
tab1, tab2 = st.tabs(["Análisis Detallado (Nivel 1 y 2)", "Matriz Global (Colección)"])

# PESTAÑA 1: Busqueda individual y comparacion
with tab1:
    c1, c2 = st.columns([2, 1])
    with c1:
        titulo_1 = st.selectbox("1. Selecciona la canción base:", df['track_title'].unique())
    with c2:
        n_similares = st.slider("2. ¿Cuántas similares buscar?", 1, 10, 5)

    # Logica de calculo
    # Localiza la cancion seleccionada y la compara matematicamente con el resto
    idx_1 = df[df['track_title'] == titulo_1].index[0]
    texto_1 = df.iloc[idx_1]['lyric']
    vec_1 = st.session_state.embeddings[idx_1].reshape(1, -1)
    
    sims = cosine_similarity(vec_1, st.session_state.embeddings)[0]
    df['Similitud'] = sims
    
    # Ordena los resultados por similitud y excluye la cancion base
    df_sorted = df.sort_values('Similitud', ascending=False)
    df_sorted = df_sorted[df_sorted['track_title'] != titulo_1] 

    st.divider()

    # Muestra los resultados principales en una tabla
    st.subheader(f"Top {n_similares} canciones más parecidas a '{titulo_1}':")
    top_n = df_sorted.head(n_similares)

    st.dataframe(
        top_n[['track_title', 'album', 'Similitud']].style.format({'Similitud': '{:.2%}'}),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # Seccion para la seleccion detallada de textos
    st.subheader("Comparación Detallada")
    col_sel, col_btn = st.columns([3, 1])

    with col_sel:
        opciones = top_n['track_title'].tolist()
        titulo_2 = st.selectbox("Elige una de la lista de arriba para comparar:", opciones)
        
        # Obtiene la letra de la segunda cancion para la comparacion
        idx_2 = df[df['track_title'] == titulo_2].index[0]
        texto_2 = df.iloc[idx_2]['lyric']

    # Visualizacion de las letras lado a lado
    col_txt1, col_txt2 = st.columns(2)

    with col_txt1:
        st.markdown(f"**- {titulo_1}** (Base)")
        st.text_area("Letra 1", texto_1, height=300, label_visibility="collapsed")

    with col_txt2:
        st.markdown(f"**- {titulo_2}** (Comparación)")
        st.text_area("Letra 2", texto_2, height=300, label_visibility="collapsed")

    # Generacion del mapa de calor
    st.markdown("<br>", unsafe_allow_html=True)    
    if st.button("Generar Mapa de Calor (Frase a Frase)", use_container_width=True, type="primary"):
        
        # Divide las letras en listas de frases
        frases_1 = split_text(texto_1)
        frases_2 = split_text(texto_2)
        
        if len(frases_1) > 0 and len(frases_2) > 0:
            # Calcula la similitud entre cada par de frases
            emb_1 = model.encode(frases_1)
            emb_2 = model.encode(frases_2)
            matriz = cosine_similarity(emb_1, emb_2)
            
            # Configura las columnas para centrar el grafico
            c_espacio, c_mapa = st.columns([1, 6]) 
            
            with c_mapa:
                fig = px.imshow(
                    matriz,
                    x=[f"T2: {f}" for f in frases_2],
                    y=[f"T1: {f}" for f in frases_1],
                    color_continuous_scale="RdBu_r",
                    aspect="equal",
                    title=f"Similitud: {titulo_1} vs. {titulo_2}"
                )
                
                # Ajustes visuales: define tamaño fijo y margenes
                fig.update_layout(
                    autosize=False,
                    width=850,
                    height=850,

                    title_x=0.6,
                    title_xanchor='center',
                    title_y=0.98,

                    margin=dict(t=60, b=60, l=100, r=100),

                    xaxis=dict(showticklabels=True, side='bottom'),
                    yaxis=dict(showticklabels=True),

                    coloraxis_colorbar=dict(
                        len=0.82,       
                        x=1.1,         
                        xpad=0,        
                        xanchor="left",
                        y=0.5,
                        yanchor="middle",
                        title="Similitud"
                    )
                )
                
                fig.update_traces(hovertemplate="<b>Frase T2:</b> %{x}<br><b>Frase T1:</b> %{y}<br><b>Similitud:</b> %{z:.2f}<extra></extra>")

                st.plotly_chart(fig, use_container_width=False)
                
        else:
            st.warning("Letras insuficientes para comparar.")

# PESTAÑA 2: Matriz Global
with tab2:
    st.header("Matriz de Similitud Global (Colección Completa)")
    
    st.markdown("""
    Esta matriz cruza **todas las canciones contra todas**.
    * Usa el **zoom** y el **cursor** sobre la matriz para identificar pares interesantes.
    * Luego selecciónalos abajo para compararlos.
    """)
    
    # Mantiene el estado del mapa activo tras la interaccion del usuario
    if 'mostrar_matriz_global' not in st.session_state:
        st.session_state.mostrar_matriz_global = False

    if st.button("Generar Matriz Global"):
        st.session_state.mostrar_matriz_global = True

    if st.session_state.mostrar_matriz_global:
        
        with st.spinner("Calculando interacciones..."):
            matriz_global = cosine_similarity(st.session_state.embeddings)
            nombres = df['track_title'].tolist()
            
            fig_g = px.imshow(
                matriz_global,
                x=nombres,
                y=nombres,
                color_continuous_scale="Viridis",
            )
            
            # Ajuste de diseño especifico para la visualizacion global
            fig_g.update_layout(
                autosize=True,
                height=1000,
                margin=dict(l=80, r=50, t=50, b=50), 

                coloraxis_colorbar=dict(
                    len=0.95,
                    x=0.75,         
                    xpad=0,       
                    xanchor="left",
                    y=0.5,
                    yanchor="middle",
                    title="Similitud"
                )
            )
            
            fig_g.update_traces(
                hovertemplate="<b>Canción A:</b> %{y}<br><b>Canción B:</b> %{x}<br><b>Similitud:</b> %{z:.2f}<extra></extra>"
            )

            st.plotly_chart(fig_g, use_container_width=True)
            
            st.divider()
            
            # Seccion de comparacion puntual para pares seleccionados
            st.subheader("Comparar par seleccionado")
            st.info("Selecciona aquí las dos canciones que viste en la matriz:")
            
            col_sel_g1, col_sel_g2 = st.columns(2)
            
            with col_sel_g1:
                sel_global_1 = st.selectbox("Canción A", df['track_title'].unique(), key="sel_g1")
                txt_g1 = df[df['track_title'] == sel_global_1]['lyric'].values[0]
                st.text_area("Texto A", txt_g1, height=200, key="txt_g1")
                
            with col_sel_g2:
                sel_global_2 = st.selectbox("Canción B", df['track_title'].unique(), key="sel_g2")
                txt_g2 = df[df['track_title'] == sel_global_2]['lyric'].values[0]
                st.text_area("Texto B", txt_g2, height=200, key="txt_g2")
            
            # Calculo y muestra del porcentaje de similitud final
            emb_g1 = st.session_state.embeddings[df[df['track_title'] == sel_global_1].index[0]]
            emb_g2 = st.session_state.embeddings[df[df['track_title'] == sel_global_2].index[0]]
            sim_score = cosine_similarity([emb_g1], [emb_g2])[0][0]
            
            st.success(f"Similitud Global entre '{sel_global_1}' y '{sel_global_2}': **{sim_score:.2%}**")