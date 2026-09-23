"""
app.py
======
Interface Web Interativa Acadêmica e Profissional (Streamlit)
Projeto: "Roteamento de Drones de Entrega com Estações de Recarga via Algoritmos Genéticos (DEAP)"
Design: Protocolo Minimalist-UI (Clean Editorial, Warm/Neutral Palette, Zero Emojis).
"""

import streamlit as st
import pandas as pd
import numpy as np
import math

from src.problem_model import DroneProblem, Depot, Customer, RechargeStation
from src.ga_engine import GeneticAlgorithmSolver
from src.visualizer import plot_route, plot_convergence

# ==========================================
# CONFIGURAÇÃO GERAL DA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Roteamento de Drones com Estações de Recarga",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# DESIGN SYSTEM: CSS MINIMALIST-UI
# ==========================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=Plus+Jakarta+Sans:wght@500;600;700&display=swap');

    /* Variáveis do Protocolo Minimalist-UI */
    :root {
        --bg-canvas: #FAFAF9;
        --bg-surface: #FFFFFF;
        --border-color: #EAEAEA;
        --border-subtle: #F1F1EF;
        --text-primary: #18181B;
        --text-secondary: #71717A;
        --text-muted: #A1A1AA;
        --accent-dark: #18181B;
        --pastel-green-bg: #EDF3EC;
        --pastel-green-txt: #2E6534;
        --pastel-red-bg: #FDEBEC;
        --pastel-red-txt: #9F2F2D;
        --pastel-blue-bg: #E1F3FE;
        --pastel-blue-txt: #1F6C9F;
        --font-mono: 'Geist Mono', 'SF Mono', monospace;
    }

    /* Fundo da Aplicação */
    .stApp {
        background-color: var(--bg-canvas);
        font-family: 'Inter', -apple-system, sans-serif;
        color: var(--text-primary);
    }

    /* Barra Lateral Minimalista */
    section[data-testid="stSidebar"] {
        background-color: #F7F6F3 !important;
        border-right: 1px solid var(--border-color);
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 600;
        letter-spacing: -0.01em;
        color: var(--text-primary);
    }

    /* Cabeçalho do Projeto */
    .header-container {
        padding: 1rem 0 1.5rem 0;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 1.5rem;
    }
    
    .authors-container {
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
        margin-bottom: 1rem;
    }
    .authors-title {
        font-family: var(--font-mono);
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .authors-list {
        font-size: 0.88rem;
        color: var(--text-primary);
        font-weight: 500;
        line-height: 1.4;
    }

    .project-title {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: var(--text-primary);
        line-height: 1.2;
        margin: 0 0 0.5rem 0;
    }
    .project-desc {
        font-size: 0.95rem;
        color: var(--text-secondary);
        line-height: 1.6;
        max-width: 90ch;
    }

    /* Cartões de Telemetria / Métricas (Flat Bento) */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.8rem;
    }
    .metric-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 1rem 1.1rem;
        transition: border-color 0.15s ease;
    }
    .metric-card:hover {
        border-color: #D4D4D8;
    }
    .metric-label {
        font-family: var(--font-mono);
        font-size: 0.70rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-secondary);
        margin-bottom: 0.35rem;
    }
    .metric-value {
        font-family: var(--font-mono);
        font-size: 1.55rem;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
        color: var(--text-primary);
        line-height: 1.1;
    }
    .metric-note {
        font-size: 0.78rem;
        color: var(--text-muted);
        margin-top: 0.35rem;
    }

    /* Banner de Status Operacional */
    .status-banner-ok {
        background: var(--pastel-green-bg);
        border: 1px solid rgba(46, 101, 52, 0.2);
        color: var(--pastel-green-txt);
        padding: 0.75rem 1rem;
        border-radius: 6px;
        font-size: 0.88rem;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }
    .status-banner-error {
        background: var(--pastel-red-bg);
        border: 1px solid rgba(159, 47, 45, 0.2);
        color: var(--pastel-red-txt);
        padding: 0.75rem 1rem;
        border-radius: 6px;
        font-size: 0.88rem;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }

    /* Botões Primários e Secundários */
    div.stButton > button {
        border-radius: 4px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        transition: all 0.15s ease !important;
        box-shadow: none !important;
    }
    div.stButton > button[kind="primary"] {
        background-color: var(--accent-dark) !important;
        color: #FFFFFF !important;
        border: 1px solid var(--accent-dark) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #27272A !important;
    }
    div.stButton > button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }
    div.stButton > button:active {
        transform: scale(0.98) !important;
    }

    /* Abas Minimalistas */
    button[data-baseweb="tab"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.90rem !important;
        color: var(--text-secondary) !important;
        padding: 0.6rem 1rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        border-bottom-color: var(--text-primary) !important;
    }

    /* Tabela de Dados */
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border-color);
        border-radius: 6px;
        background: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_scenario(num_customers: int, num_stations: int, seed: int) -> dict:
    """Gera e armazena nós fixos no session_state para manter estabilidade visual."""
    rng = np.random.default_rng(seed)
    area_size = 30.0

    depot = Depot(id=0, name="Base Central", x=0.0, y=0.0)

    customers = []
    for i in range(num_customers):
        cx = float(rng.uniform(-area_size / 2, area_size / 2))
        cy = float(rng.uniform(-area_size / 2, area_size / 2))
        customers.append(
            Customer(id=i + 1, name=f"Cliente {i + 1}", x=round(cx, 2), y=round(cy, 2))
        )

    stations = []
    for j in range(num_stations):
        angle = (2 * math.pi * j / max(1, num_stations)) + float(rng.uniform(-0.25, 0.25))
        radius = float(rng.uniform(area_size * 0.25, area_size * 0.42))
        sx = float(radius * math.cos(angle))
        sy = float(radius * math.sin(angle))
        stations.append(
            RechargeStation(
                id=num_customers + j + 1,
                name=f"Estacao {j + 1}",
                x=round(sx, 2),
                y=round(sy, 2),
            )
        )

    return {
        "depot": depot,
        "customers": customers,
        "stations": stations,
        "num_customers": num_customers,
        "num_stations": num_stations,
        "seed": seed,
    }


# ==========================================
# BARRA LATERAL (CONTROLES DO SISTEMA)
# ==========================================
st.sidebar.markdown(
    """
    <div style="padding-bottom: 0.5rem; border-bottom: 1px solid #EAEAEA; margin-bottom: 1rem;">
        <span style="font-family: 'Geist Mono', monospace; font-size: 0.70rem; color: #71717A; text-transform: uppercase; letter-spacing: 0.08em;">Parametros Operacionais</span>
        <h3 style="margin: 0.2rem 0 0 0; font-size: 1.1rem; color: #18181B;">Configuracao do Problema</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

num_customers = st.sidebar.slider("Clientes de Entrega", min_value=5, max_value=30, value=12, step=1)
num_stations = st.sidebar.slider("Estacoes de Recarga", min_value=1, max_value=5, value=2, step=1)
battery_capacity = st.sidebar.slider("Autonomia da Bateria (km)", min_value=10, max_value=60, value=25, step=1)

seed_input = st.sidebar.number_input("Semente Aleatoria (Seed)", min_value=0, max_value=99999, value=42, step=1)

col_b1, col_b2 = st.sidebar.columns(2)
new_scenario_clicked = col_b1.button("Novo Cenario", use_container_width=True)

st.sidebar.markdown(
    """
    <div style="padding: 1rem 0 0.5rem 0; border-top: 1px solid #EAEAEA; margin-top: 1rem;">
        <span style="font-family: 'Geist Mono', monospace; font-size: 0.70rem; color: #71717A; text-transform: uppercase; letter-spacing: 0.08em;">Algoritmo Genetico</span>
        <h3 style="margin: 0.2rem 0 0 0; font-size: 1.1rem; color: #18181B;">Hiperparametros GA</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

pop_size = st.sidebar.slider("Tamanho da Populacao", min_value=20, max_value=200, value=50, step=5)
generations = st.sidebar.slider("Numero de Geracoes", min_value=10, max_value=200, value=50, step=5)
crossover_prob = st.sidebar.slider("Taxa de Cruzamento (OX1)", min_value=0.50, max_value=0.95, value=0.80, step=0.05)
mutation_prob = st.sidebar.slider("Taxa de Mutacao (Shuffle)", min_value=0.01, max_value=0.30, value=0.10, step=0.01)

run_ga_clicked = st.sidebar.button("Executar Otimizacao", type="primary", use_container_width=True)


# ==========================================
# GESTÃO DE ESTADO (st.session_state)
# ==========================================
if "scenario" not in st.session_state:
    st.session_state.scenario = init_scenario(num_customers, num_stations, seed_input)
    st.session_state.seed_counter = seed_input

if new_scenario_clicked:
    st.session_state.seed_counter += 1
    st.session_state.scenario = init_scenario(num_customers, num_stations, st.session_state.seed_counter)
    if "last_result" in st.session_state:
        del st.session_state["last_result"]
elif (
    st.session_state.scenario["num_customers"] != num_customers
    or st.session_state.scenario["num_stations"] != num_stations
    or (not new_scenario_clicked and st.session_state.scenario["seed"] != seed_input)
):
    st.session_state.scenario = init_scenario(num_customers, num_stations, seed_input)
    if "last_result" in st.session_state:
        del st.session_state["last_result"]

current_scenario = st.session_state.scenario
problem = DroneProblem(
    depot=current_scenario["depot"],
    customers=current_scenario["customers"],
    recharge_stations=current_scenario["stations"],
    battery_capacity=battery_capacity,
    consumption_rate=1.0,
    recharge_penalty_time=15.0,
    battery_penalty=100000.0,
)

if "last_result" not in st.session_state or run_ga_clicked:
    with st.spinner("Processando convergencia do Algoritmo Genetico..."):
        solver = GeneticAlgorithmSolver(
            problem=problem,
            population_size=pop_size,
            generations=generations,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            tournament_size=3,
            elitism_size=2,
            seed=st.session_state.scenario["seed"],
        )
        best_ind, best_res, history, elapsed_time = solver.solve()

        st.session_state.last_result = {
            "best_individual": best_ind,
            "route_result": best_res,
            "history": history,
            "elapsed_time": elapsed_time,
            "battery_capacity": battery_capacity,
        }

res = st.session_state.last_result
route_res = res["route_result"]
history = res["history"]
elapsed_time = res["elapsed_time"]


# ==========================================
# PAINEL PRINCIPAL
# ==========================================

# 1. Cabeçalho com identificação dos alunos em ordem alfabética
st.markdown(
    """
    <div class="header-container">
        <div class="authors-container">
            <span class="authors-title">Alunos:</span>
            <div class="authors-list">
                Luiz Gustavo Guaycurus Goulart<br>
                Matheus Corrêa Cesar Biancovilli<br>
                Nicholas Pompilio Coda
            </div>
        </div>
        <h1 class="project-title">Roteamento de Drones com Estações de Recarga</h1>
        <div class="project-desc">
            Otimização de rotas aéreas autônomas sob restrições de bateria, política de refúgio preventivo 
            (Lookahead) e inserção de paradas de recarga via Algoritmos Genéticos (DEAP).
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 2. Banner de Status da Operação
if route_res.is_feasible:
    st.markdown(
        f"""
        <div class="status-banner-ok">
            <strong>Operacao Concluida com Sucesso</strong>: Todas as {len(problem.customers)} entregas foram atendidas e a frota retornou a Base Central com reserva de energia.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class="status-banner-error">
            <strong>Inviabilidade Energetica Detectada</strong>: A autonomia atual de {battery_capacity} km foi insuficiente para o trajeto, ocasionando {route_res.crashes} queda(s) por esgotamento de bateria. Aumente a autonomia ou adicione mais estacoes de recarga.
        </div>
        """,
        unsafe_allow_html=True,
    )

# 3. Métricas Principais (Bento Minimalista)
status_str = "Pouso Seguro" if route_res.is_feasible else f"Queda ({route_res.crashes})"

st.markdown(
    f"""
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Distancia Total</div>
            <div class="metric-value">{route_res.total_distance:.1f} <span style="font-size: 0.85rem; color: #71717A;">km</span></div>
            <div class="metric-note">Trajetoria percorrida</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Paradas de Recarga</div>
            <div class="metric-value">{route_res.recharge_count}</div>
            <div class="metric-note">Estacoes utilizadas</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Status da Bateria</div>
            <div class="metric-value" style="font-size: 1.35rem;">{status_str}</div>
            <div class="metric-note">Condicao final</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Tempo de Convergencia</div>
            <div class="metric-value">{elapsed_time * 1000:.0f} <span style="font-size: 0.85rem; color: #71717A;">ms</span></div>
            <div class="metric-note">{generations} geracoes · {pop_size} ind.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 4. Abas de Conteúdo
tab1, tab2, tab3, tab4 = st.tabs([
    "Plano de Rota 2D",
    "Curva de Convergência",
    "Tabela de Telemetria",
    "Metodologia e Formulação",
])

with tab1:
    fig_map = plot_route(problem, route_res)
    st.plotly_chart(fig_map, use_container_width=True)

with tab2:
    fig_conv = plot_convergence(history)
    st.plotly_chart(fig_conv, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    best_f = history['min_fitness'][-1]
    init_f = history['min_fitness'][0]
    gain = ((init_f - best_f) / init_f * 100) if init_f > 0 else 0.0

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Fitness Inicial (Geracao 0)</div>
                <div class="metric-value" style="font-size: 1.25rem;">{init_f:.1f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Melhor Fitness (Geracao {generations})</div>
                <div class="metric-value" style="font-size: 1.25rem;">{best_f:.1f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Reducao do Custo</div>
                <div class="metric-value" style="font-size: 1.25rem;">-{gain:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab3:
    st.markdown(
        """
        <div style="margin-bottom: 0.8rem;">
            <h3 style="margin: 0; font-size: 1.1rem; color: #18181B;">Balanço Energético por Segmento de Voo</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    table_data = []
    for leg in route_res.legs:
        status_tag = "Queda" if leg.crashed else ("Recarga" if leg.is_recharge else "Nominal")
        table_data.append({
            "Seq": f"#{leg.leg_index:02d}",
            "Origem": leg.from_node.name,
            "Destino": leg.to_node.name,
            "Distancia (km)": f"{leg.distance:.2f}",
            "Bateria Inicial (km)": f"{leg.battery_before:.2f}",
            "Bateria Final (km)": f"{leg.battery_after:.2f}",
            "Carga Residual": f"{leg.battery_pct_after:.1f}%",
            "Condicao": status_tag,
            "Descricao": leg.event_description,
        })

    df_legs = pd.DataFrame(table_data)
    st.dataframe(df_legs, use_container_width=True, hide_index=True)

with tab4:
    st.markdown(
        """
        ### Fundamentação Teórica e Formulação Matemática

        #### 1. Codificação do Indivíduo (Cromossomo)
        - O indivíduo é modelado como uma **permutação dos índices dos clientes** $\\pi = (c_1, c_2, \\dots, c_N)$, garantindo que cada cliente seja visitado exatamente uma vez.
        - As estações de recarga não ocupam posições fixas no cromossomo. Um algoritmo com política de **Lookahead** insere dinamicamente a estação mais eficiente no trajeto quando a bateria restante não permite atender o próximo cliente com margem de segurança.

        #### 2. Função de Aptidão (Fitness)
        Objetivo de **minimização**:
        $$\\min_{\\pi} \\quad \\mathcal{F}(\\pi) = \\sum_{k=1}^{M} d(v_{k-1}, v_k) + \\lambda_{\\text{rec}} \\cdot R(\\pi) + \\lambda_{\\text{queda}} \\cdot K(\\pi)$$
        Onde:
        - $d(v_{k-1}, v_k)$: Distância euclidiana percorrida em cada trecho.
        - $R(\\pi)$: Quantidade total de paradas em estações de recarga.
        - $\\lambda_{\\text{rec}}$: Custo de penalidade por recarga ($15.0$ km equivalentes).
        - $K(\\pi)$: Número de quedas por esgotamento de bateria antes de atingir um refúgio.
        - $\\lambda_{\\text{queda}}$: Penalidade severa por inviabilidade ($100.000$).

        #### 3. Operadores Genéticos (DEAP)
        - **Seleção:** Torneio ($k=3$) para balancear convergência e diversidade.
        - **Cruzamento:** Order Crossover (**OX1**), preservando a integridade da permutação.
        - **Mutação:** Shuffle Indexes (embaralhamento de índices).
        - **Elitismo:** Hall of Fame preservando as duas melhores soluções entre as gerações.
        """
    )
