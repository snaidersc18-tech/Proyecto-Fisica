import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA Y ESTILOS
# ==========================================
st.set_page_config(
    page_title="Simulador Avanzado: Plano Doble",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para la estética Dark/Neon
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: transparent;
        border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 243, 255, 0.1); border-bottom: 2px solid #00f3ff; color: #00f3ff !important;
    }
    .fuerza-box { padding: 15px; border-radius: 8px; background-color: #1e1e2e; margin-bottom: 15px; border-left: 4px solid; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MOTOR DE SIMULACIÓN FÍSICA Y CINEMÁTICA
# ==========================================
def calcular_dinamica(m1, m2, th1_deg, th2_deg, mu_s, mu_k, conexion, k, g, dt=0.02, t_max=4.0):
    th1 = np.radians(th1_deg)
    th2 = np.radians(th2_deg)
    
    # Fuerzas de gravedad paralelas y perpendiculares al plano
    W1x = m1 * g * np.sin(th1) # Positivo = hacia abajo de la rampa
    W1y = m1 * g * np.cos(th1)
    N1 = W1y
    
    W2x = m2 * g * np.sin(th2) # Positivo = hacia abajo de la rampa
    W2y = m2 * g * np.cos(th2)
    N2 = W2y
    
    fs1_max, fk1 = mu_s * N1, mu_k * N1
    fs2_max, fk2 = mu_s * N2, mu_k * N2
    
    # Geometría del plano
    H = 10.0
    hyp1 = H / np.sin(th1)
    hyp2 = H / np.sin(th2)
    
    # Tamaño de los bloques (para límites físicos en la animación)
    w1 = 2.0 + (m1 * 0.03)
    w2 = 2.0 + (m2 * 0.03)
    
    # Posición 's' es la distancia desde la polea hacia abajo por la rampa
    s1, s2 = hyp1 * 0.5, hyp2 * 0.5 
    L0_resorte = s1 + s2
    
    v1, v2 = 0.0, 0.0
    tiempos = np.arange(0, t_max, dt)
    hist_s1, hist_s2, hist_v1, hist_v2, hist_T = [], [], [], [], []
    
    a_sis, T_estatico = 0.0, 0.0 # Variables para mostrar estado inicial
    
    for i, t in enumerate(tiempos):
        T_val = 0.0
        if conexion == "Cuerda Rígida":
            # v1 positivo significa que m1 baja y m2 sube
            F_drive = W1x - W2x
            fric_est = fs1_max + fs2_max
            
            if abs(v1) < 1e-4 and abs(F_drive) <= fric_est:
                a_q = 0.0
                v1 = 0.0
                # Tensión estática que equilibra el sistema
                T_val = W1x if F_drive > 0 else W2x
            else:
                dir_mov = np.sign(v1) if abs(v1) > 1e-4 else np.sign(F_drive)
                fric_cin = dir_mov * (fk1 + fk2)
                
                a_q = (F_drive - fric_cin) / (m1 + m2)
                v1 += a_q * dt
                T_val = W1x - m1 * a_q - np.sign(v1) * fk1
            
            if i == 0: a_sis, T_estatico = a_q, T_val
                
            v2 = -v1
            s1 += v1 * dt
            s2 += v2 * dt
            
        else: # Sistema con Resorte o Masas Libres
            if conexion == "Resorte":
                x_ext = (s1 + s2) - L0_resorte
                T_val = k * x_ext # Fuerza elástica de Hooke
            
            if i == 0: T_estatico = T_val
                
            # Dinámica Masa 1
            F_net1 = W1x - T_val
            if abs(v1) < 1e-4 and abs(F_net1) <= fs1_max:
                a1 = 0.0
                v1 = 0.0
            else:
                dir_v1 = np.sign(v1) if abs(v1) > 1e-4 else np.sign(F_net1)
                a1 = (F_net1 - dir_v1 * fk1) / m1
                v1 += a1 * dt
            s1 += v1 * dt
            if i == 0 and conexion != "Cuerda Rígida": a_sis = (a1, 0)
            
            # Dinámica Masa 2
            F_net2 = W2x - T_val
            if abs(v2) < 1e-4 and abs(F_net2) <= fs2_max:
                a2 = 0.0
                v2 = 0.0
            else:
                dir_v2 = np.sign(v2) if abs(v2) > 1e-4 else np.sign(F_net2)
                a2 = (F_net2 - dir_v2 * fk2) / m2
                v2 += a2 * dt
            s2 += v2 * dt
            if i == 0 and conexion != "Cuerda Rígida": a_sis = (a1, a2)

        # Restricciones físicas: Evitar que se caigan de la rampa o atraviesen la polea
        s1 = np.clip(s1, w1/2, hyp1 - w1/2)
        s2 = np.clip(s2, w2/2, hyp2 - w2/2)
        if s1 in (w1/2, hyp1 - w1/2): v1 = 0
        if s2 in (w2/2, hyp2 - w2/2): v2 = 0
        
        hist_s1.append(s1); hist_s2.append(s2)
        hist_v1.append(v1); hist_v2.append(v2)
        hist_T.append(T_val)

    estado_fuerzas = {
        'W1x': W1x, 'W1y': W1y, 'N1': N1, 'fs1': fs1_max, 'fk1': fk1,
        'W2x': W2x, 'W2y': W2y, 'N2': N2, 'fs2': fs2_max, 'fk2': fk2,
        'a': a_sis, 'T': T_estatico
    }
        
    return tiempos, np.array(hist_s1), np.array(hist_s2), np.array(hist_v1), np.array(hist_v2), np.array(hist_T), estado_fuerzas

# ==========================================
# 3. RENDERIZADO GRÁFICO (ANIMACIÓN CORREGIDA)
# ==========================================
def crear_animacion(t, s1_arr, s2_arr, th1_deg, th2_deg, m1, m2, conexion):
    H = 10.0 
    th1 = np.radians(th1_deg)
    th2 = np.radians(th2_deg)
    
    L1 = H / np.tan(th1)
    L2 = H / np.tan(th2)
    
    w1, h1 = 2.0 + (m1*0.03), 2.0 + (m1*0.03)
    w2, h2 = 2.0 + (m2*0.03), 2.0 + (m2*0.03)
    
    def pos_bloque(s, lado, w, h, th_rad):
        # Transforma la distancia 's' en coordenadas X, Y rotadas
        if lado == 1:
            xc = -s * np.cos(th_rad)
            yc = H - s * np.sin(th_rad)
            angulo = th_rad # Inclinación izquierda
        else:
            xc = s * np.cos(th_rad)
            yc = H - s * np.sin(th_rad)
            angulo = -th_rad # Inclinación derecha
            
        esquinas = np.array([[-w/2, 0], [w/2, 0], [w/2, h], [-w/2, h]])
        rot = np.array([[np.cos(angulo), -np.sin(angulo)], [np.sin(angulo), np.cos(angulo)]])
        rotadas = np.dot(esquinas, rot.T)
        
        # Coordenada superior del bloque para anclar la cuerda
        cx_top = xc - h * np.sin(angulo)
        cy_top = yc + h * np.cos(angulo)
        
        return np.append(rotadas[:,0]+xc, rotadas[0,0]+xc), np.append(rotadas[:,1]+yc, rotadas[0,1]+yc), cx_top, cy_top

    fig = go.Figure()

    # Trazar Cuña estática
    fig.add_trace(go.Scatter(x=[-L1, 0, L2, -L1], y=[0, H, 0, 0], fill="toself", fillcolor="#1a1a2e", line=dict(color="#00f3ff", width=2), hoverinfo="skip"))
    
    # Trazar Polea
    H_polea = H + max(h1, h2)/2
    fig.add_trace(go.Scatter(x=[0], y=[H_polea], mode="markers", marker=dict(size=25, color="#e94560", line=dict(width=3, color="white")), hoverinfo="skip"))

    # Configurar Frame Inicial (0)
    bx1, by1, cx1, cy1 = pos_bloque(s1_arr[0], 1, w1, h1, th1)
    bx2, by2, cx2, cy2 = pos_bloque(s2_arr[0], 2, w2, h2, th2)
    
    t_cuerda = go.Scatter(x=[cx1, 0, cx2], y=[cy1, H_polea, cy2], mode="lines", line=dict(color="white", width=2, dash="dot" if conexion == "Resorte" else "solid"), visible=(conexion != "Desconectados"))
    t_m1 = go.Scatter(x=bx1, y=by1, fill="toself", fillcolor="#00f3ff", line=dict(color="white"), name="m1")
    t_m2 = go.Scatter(x=bx2, y=by2, fill="toself", fillcolor="#b800ff", line=dict(color="white"), name="m2")

    fig.add_traces([t_cuerda, t_m1, t_m2])

    # Generación de Frames para la animación
    frames = []
    step = max(1, len(t) // 60) # Optimizar a ~60 fps para evitar lag en Streamlit
    for i in range(0, len(t), step):
        b1_x, b1_y, c1_x, c1_y = pos_bloque(s1_arr[i], 1, w1, h1, th1)
        b2_x, b2_y, c2_x, c2_y = pos_bloque(s2_arr[i], 2, w2, h2, th2)
        frames.append(go.Frame(data=[
            go.Scatter(x=[c1_x, 0, c2_x], y=[c1_y, H_polea, c2_y]),
            go.Scatter(x=b1_x, y=b1_y),
            go.Scatter(x=b2_x, y=b2_y)
        ], name=f"fr{i}"))
        
    fig.frames = frames

    # EL SECRETO DE ASPECTO: scaleanchor="x" y scaleratio=1 previenen que los cuadrados se conviertan en rombos
    fig.update_layout(
        xaxis=dict(range=[-(L1+5), L2+5], visible=False),
        yaxis=dict(scaleanchor="x", scaleratio=1, range=[-2, H+6], visible=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0), showlegend=False, height=500,
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="▶ Reproducir Animación", method="animate", args=[None, dict(frame=dict(duration=30, redraw=True), fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pausa / Reset", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=True), mode="immediate")])
            ], direction="left", pad={"r": 10, "t": 87}, showactive=False, x=0.5, xanchor="center", y=-0.1, yanchor="top", bgcolor="#1a1a2e", font=dict(color="#00f3ff")
        )]
    )
    return fig

# ==========================================
# 4. INTERFAZ Y BARRA LATERAL (UI/UX)
# ==========================================
st.title("🌌 Simulador Cinemático: Plano Doble")

with st.sidebar:
    st.header("⚙️ Parámetros del Sistema")
    conexion = st.radio("🔗 Tipo de Conexión", ["Cuerda Rígida", "Resorte", "Desconectados"], index=0)
    k_resorte = st.slider("Constante k (N/m)", 10.0, 500.0, 100.0) if conexion == "Resorte" else 0.0
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h4 style='color:#00f3ff;'>Masa 1 (Izq)</h4>", unsafe_allow_html=True)
        m1 = st.number_input("m1 (kg)", 1.0, 100.0, 5.0)
        th1 = st.slider("θ₁ (°)", 5, 85, 30)
    with col2:
        st.markdown("<h4 style='color:#b800ff;'>Masa 2 (Der)</h4>", unsafe_allow_html=True)
        m2 = st.number_input("m2 (kg)", 1.0, 100.0, 10.0)
        th2 = st.slider("θ₂ (°)", 5, 85, 45)
    st.divider()
    
    g_val = st.number_input("Gravedad g (m/s²)", value=9.81)
    material = st.selectbox("🧊 Coeficientes Fricción", ["Personalizado", "Sin Fricción", "Madera/Madera (0.4 / 0.2)", "Acero/Hielo (0.03 / 0.01)"])
    if material == "Sin Fricción": def_s, def_k = 0.0, 0.0
    elif material == "Madera/Madera (0.4 / 0.2)": def_s, def_k = 0.4, 0.2
    elif material == "Acero/Hielo (0.03 / 0.01)": def_s, def_k = 0.03, 0.01
    else: def_s, def_k = 0.3, 0.15
    
    c1, c2 = st.columns(2)
    mu_s = c1.number_input("μ Estático", 0.0, 1.5, def_s, disabled=(material != "Personalizado"))
    mu_k = c2.number_input("μ Cinético", 0.0, 1.5, def_k, disabled=(material != "Personalizado"))

# Procesar motor físico
t_arr, s1_arr, s2_arr, v1_arr, v2_arr, T_arr, F = calcular_dinamica(m1, m2, th1, th2, mu_s, mu_k, conexion, k_resorte, g_val)

# ==========================================
# 5. PESTAÑAS PRINCIPALES
# ==========================================
tab_sim, tab_resultados, tab_graficas = st.tabs(["🎮 Animación", "📊 Desglose de Fuerzas", "📈 Gráficas"])

with tab_sim:
    st.plotly_chart(crear_animacion(t_arr, s1_arr, s2_arr, th1, th2, m1, m2, conexion), use_container_width=True)
    
    m_c1, m_c2, m_c3 = st.columns(3)
    m_c1.metric("Dist. Recorrida m1", f"{abs(s1_arr[-1] - s1_arr[0]):.2f} m")
    m_c2.metric("Dist. Recorrida m2", f"{abs(s2_arr[-1] - s2_arr[0]):.2f} m")
    m_c3.metric("Aceleración Inicial", f"{F['a']:.2f} m/s²" if not isinstance(F['a'], tuple) else f"{F['a'][0]:.2f} | {F['a'][1]:.2f} m/s²")

with tab_resultados:
    st.markdown("### Análisis Estático de Fuerzas (Diagrama de Cuerpo Libre)")
    st.info("Convención de Ejes: Eje **X** paralelo al plano (positivo hacia *abajo* de la rampa), Eje **Y** perpendicular al plano (positivo hacia *arriba*).")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(f"<div class='fuerza-box' style='border-color: #00f3ff;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#00f3ff; margin-top:0;'>Bloque 1 (Izquierdo)</h3>", unsafe_allow_html=True)
        st.latex(r"\Sigma F_y = N_1 - W_{1y} = 0")
        st.latex(r"N_1 = m_1 \cdot g \cdot \cos(\theta_1) = " + f"{F['N1']:.2f} \\text{{ N}}")
        st.divider()
        st.latex(r"W_{1x} = m_1 \cdot g \cdot \sin(\theta_1) = " + f"{F['W1x']:.2f} \\text{{ N}}")
        
        # Corrección de llaves dobles aplicadas aquí para f_k1
        st.latex(r"f_{s1(max)} = \mu_s N_1 = " + f"{F['fs1']:.2f} \\text{{ N}} \quad | \quad f_{{k1}} = \mu_k N_1 = " + f"{F['fk1']:.2f} \\text{{ N}}")
        st.divider()
        st.latex(r"\Sigma F_x = W_{1x} - T \pm f_1 = m_1 a_1")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_m2:
        st.markdown(f"<div class='fuerza-box' style='border-color: #b800ff;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#b800ff; margin-top:0;'>Bloque 2 (Derecho)</h3>", unsafe_allow_html=True)
        st.latex(r"\Sigma F_y = N_2 - W_{2y} = 0")
        st.latex(r"N_2 = m_2 \cdot g \cdot \cos(\theta_2) = " + f"{F['N2']:.2f} \\text{{ N}}")
        st.divider()
        st.latex(r"W_{2x} = m_2 \cdot g \cdot \sin(\theta_2) = " + f"{F['W2x']:.2f} \\text{{ N}}")
        
        # Corrección de llaves dobles aplicadas aquí para f_k2
        st.latex(r"f_{s2(max)} = \mu_s N_2 = " + f"{F['fs2']:.2f} \\text{{ N}} \quad | \quad f_{{k2}} = \mu_k N_2 = " + f"{F['fk2']:.2f} \\text{{ N}}")
        st.divider()
        st.latex(r"\Sigma F_x = W_{2x} - T \pm f_2 = m_2 a_2")
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("### Sistema Combinado")
    st.latex(r"\text{Tensión / Fuerza Elástica ( } T \text{ )} = " + f"{F['T']:.2f} \\text{{ N}}")
    if conexion == "Cuerda Rígida":
        st.latex(r"a_{sis} = \frac{W_{1x} - W_{2x} \pm f_k}{(m_1 + m_2)}")

with tab_graficas:
    df_graf = pd.DataFrame({'Tiempo (s)': t_arr, 'Velocidad m1': v1_arr, 'Velocidad m2': v2_arr})
    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(x=df_graf['Tiempo (s)'], y=df_graf['Velocidad m1'], line=dict(color='#00f3ff', width=3), name='Vel. m1 (Izq)'))
    if conexion != "Cuerda Rígida":
        fig_v.add_trace(go.Scatter(x=df_graf['Tiempo (s)'], y=df_graf['Velocidad m2'], line=dict(color='#b800ff', width=3), name='Vel. m2 (Der)'))
    fig_v.update_layout(title="Perfil de Velocidades", template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_v, use_container_width=True)