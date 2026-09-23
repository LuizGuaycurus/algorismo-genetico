"""
visualizer.py
=============
Visualização gráfica de alta precisão técnica com Plotly.
Diretrizes: Protocolo Minimalist-UI (Clean Editorial, White/Neutral Palette,
eliminação total de emojis e tooltips limpos sem tags HTML não suportadas).
"""

from typing import Dict, List, Any
import plotly.graph_objects as go
import numpy as np

from src.problem_model import DroneProblem, RouteResult


def plot_route(problem: DroneProblem, route_result: RouteResult) -> go.Figure:
    """
    Gera mapa 2D minimalista e de alta precisão técnica para o roteamento do drone.
    Usa paleta monocromática com acentos pontuais, tipografia limpa e
    tooltips compatíveis com a renderização SVG do Plotly (sem tags div/style).
    """
    fig = go.Figure()

    is_feasible = route_result.is_feasible
    main_flight_color = "#1E293B" if is_feasible else "#DC2626"

    # 1. Desenho da Trajetória Segmentada por Pernas de Voo
    for i, leg in enumerate(route_result.legs):
        leg_label = f"Trecho {leg.leg_index}: {leg.from_node.name} -> {leg.to_node.name}"

        # Cores e estilos por condição de voo
        if leg.crashed:
            leg_color = "#DC2626"
            leg_dash = "dash"
            status_desc = "Critico: Queda por falta de bateria"
        elif leg.is_recharge:
            leg_color = "#059669"
            leg_dash = "solid"
            status_desc = "Parada de Recarga (Autonomia restaurada a 100%)"
        else:
            leg_color = main_flight_color
            leg_dash = "solid" if is_feasible else "dot"
            status_desc = "Voo Nominal"

        # Tooltip 100% nativo do Plotly (apenas <b> e <br>, sem tags <div> com CSS)
        tooltip_text = (
            f"<b>{leg_label}</b><br>"
            f"Status: {status_desc}<br>"
            f"Distancia: {leg.distance:.2f} km<br>"
            f"Bateria Inicial: {leg.battery_before:.2f} km ({(leg.battery_before / problem.battery_capacity) * 100:.1f}%)<br>"
            f"Bateria Final: {leg.battery_after:.2f} km ({leg.battery_pct_after:.1f}%)<br>"
            f"Evento: {leg.event_description}"
        )

        fig.add_trace(
            go.Scatter(
                x=[leg.from_node.x, leg.to_node.x],
                y=[leg.from_node.y, leg.to_node.y],
                mode="lines",
                line=dict(color=leg_color, width=2.5 if not leg.crashed else 2.0, dash=leg_dash),
                hoverinfo="text",
                hovertext=tooltip_text,
                hoverlabel=dict(
                    bgcolor="#FFFFFF",
                    bordercolor=leg_color,
                    font=dict(family="SF Mono, JetBrains Mono, monospace", size=11, color="#0F172A"),
                ),
                name="Trajetoria de Voo" if i == 0 else None,
                showlegend=(i == 0),
            )
        )

        # Vetores direcionais discretos
        dx = leg.to_node.x - leg.from_node.x
        dy = leg.to_node.y - leg.from_node.y
        dist = np.hypot(dx, dy)
        if dist > 0.8:
            mid_x = (leg.from_node.x + leg.to_node.x) / 2
            mid_y = (leg.from_node.y + leg.to_node.y) / 2
            scale = min(0.9, dist * 0.22)
            fig.add_annotation(
                x=mid_x + (dx / dist) * scale,
                y=mid_y + (dy / dist) * scale,
                ax=mid_x - (dx / dist) * scale,
                ay=mid_y - (dy / dist) * scale,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.1,
                arrowwidth=1.8,
                arrowcolor=leg_color,
                opacity=0.85,
            )

    # 2. Estações de Recarga
    if problem.recharge_stations:
        st_x = [s.x for s in problem.recharge_stations]
        st_y = [s.y for s in problem.recharge_stations]

        st_hover = [
            f"<b>Estacao de Recarga: {s.name}</b><br>"
            f"Identificador: No {s.id}<br>"
            f"Coordenadas: ({s.x:.2f}, {s.y:.2f})<br>"
            f"Efeito: Restaura autonomia para 100%"
            for s in problem.recharge_stations
        ]

        fig.add_trace(
            go.Scatter(
                x=st_x,
                y=st_y,
                mode="markers+text",
                text=[s.name for s in problem.recharge_stations],
                textposition="top center",
                textfont=dict(size=10, color="#047857", family="Geist Sans, Inter, sans-serif"),
                marker=dict(
                    symbol="diamond",
                    size=14,
                    color="#059669",
                    line=dict(color="#FFFFFF", width=2),
                ),
                hoverinfo="text",
                hovertext=st_hover,
                hoverlabel=dict(
                    bgcolor="#FFFFFF",
                    bordercolor="#059669",
                    font=dict(family="SF Mono, JetBrains Mono, monospace", size=11, color="#047857"),
                ),
                name="Estacoes de Recarga",
            )
        )

    # 3. Clientes de Entrega
    if problem.customers:
        cust_x = [c.x for c in problem.customers]
        cust_y = [c.y for c in problem.customers]

        cust_hover = [
            f"<b>Cliente {c.id}</b> ({c.name})<br>"
            f"Coordenadas: ({c.x:.2f}, {c.y:.2f})<br>"
            f"Demanda: {c.demand:.1f} pacote"
            for c in problem.customers
        ]

        fig.add_trace(
            go.Scatter(
                x=cust_x,
                y=cust_y,
                mode="markers+text",
                text=[f"C{c.id}" for c in problem.customers],
                textposition="bottom center",
                textfont=dict(size=9, color="#9F1239", family="SF Mono, JetBrains Mono, monospace"),
                marker=dict(
                    symbol="circle",
                    size=11,
                    color="#E11D48",
                    line=dict(color="#FFFFFF", width=1.8),
                ),
                hoverinfo="text",
                hovertext=cust_hover,
                hoverlabel=dict(
                    bgcolor="#FFFFFF",
                    bordercolor="#E11D48",
                    font=dict(family="SF Mono, JetBrains Mono, monospace", size=11, color="#9F1239"),
                ),
                name="Clientes Obrigatorios",
            )
        )

    # 4. Base Central (Depósito)
    depot = problem.depot
    depot_hover = (
        f"<b>Base Central (Deposito)</b><br>"
        f"Origem e retorno final da frota<br>"
        f"Coordenadas: ({depot.x:.2f}, {depot.y:.2f})<br>"
        f"Autonomia da Frota: {problem.battery_capacity:.1f} km"
    )

    fig.add_trace(
        go.Scatter(
            x=[depot.x],
            y=[depot.y],
            mode="markers+text",
            text=["Base Central"],
            textposition="top right",
            textfont=dict(size=10, color="#0F172A", family="Geist Sans, Inter, sans-serif"),
            marker=dict(
                symbol="square",
                size=16,
                color="#0F172A",
                line=dict(color="#FFFFFF", width=2),
            ),
            hoverinfo="text",
            hovertext=depot_hover,
            hoverlabel=dict(
                bgcolor="#FFFFFF",
                bordercolor="#0F172A",
                font=dict(family="SF Mono, JetBrains Mono, monospace", size=11, color="#0F172A"),
            ),
            name="Base Central (Deposito)",
        )
    )

    # Layout Minimalista (Clean Editorial)
    status_header = (
        "Status: Rota Viavel com Pouso Seguro"
        if is_feasible
        else f"Status: Rota Inviavel ({route_result.crashes} queda(s) por falta de bateria)"
    )
    status_color = "#059669" if is_feasible else "#DC2626"

    fig.update_layout(
        title=dict(
            text=f"<span style='font-size: 13px; font-weight: 600; color: {status_color};'>{status_header}</span>",
            x=0.02,
            y=0.96,
        ),
        xaxis=dict(
            title=dict(
                text="Coordenada X (km)",
                font=dict(family="Geist Sans, Inter, sans-serif", size=11, color="#64748B"),
            ),
            gridcolor="#F1F5F9",
            gridwidth=1,
            zerolinecolor="#CBD5E1",
            zerolinewidth=1,
            scaleanchor="y",
            scaleratio=1,
            tickfont=dict(family="SF Mono, JetBrains Mono, monospace", size=10, color="#64748B"),
        ),
        yaxis=dict(
            title=dict(
                text="Coordenada Y (km)",
                font=dict(family="Geist Sans, Inter, sans-serif", size=11, color="#64748B"),
            ),
            gridcolor="#F1F5F9",
            gridwidth=1,
            zerolinecolor="#CBD5E1",
            zerolinewidth=1,
            tickfont=dict(family="SF Mono, JetBrains Mono, monospace", size=10, color="#64748B"),
        ),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FFFFFF",
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            bgcolor="#FFFFFF",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(family="Geist Sans, Inter, sans-serif", size=11, color="#334155"),
        ),
        margin=dict(l=30, r=30, t=50, b=50),
    )

    return fig


def plot_convergence(history: Dict[str, List[Any]]) -> go.Figure:
    """
    Gera gráfico de convergência geracional limpo e editorial.
    """
    fig = go.Figure()

    generations = history.get("generation", [])
    min_fitness = history.get("min_fitness", [])
    avg_fitness = history.get("avg_fitness", [])

    # Média da População
    fig.add_trace(
        go.Scatter(
            x=generations,
            y=avg_fitness,
            mode="lines",
            name="Fitness Medio da Populacao",
            line=dict(color="#94A3B8", width=1.5, dash="dash"),
            hovertemplate="Geracao %{x}<br>Media: %{y:.1f}<extra></extra>",
        )
    )

    # Melhor Fitness (Mínimo)
    fig.add_trace(
        go.Scatter(
            x=generations,
            y=min_fitness,
            mode="lines+markers",
            name="Melhor Fitness (Elite)",
            line=dict(color="#0F172A", width=2.5),
            marker=dict(size=4, color="#0F172A"),
            hovertemplate="Geracao %{x}<br><b>Melhor: %{y:.1f}</b><extra></extra>",
        )
    )

    best_val = min(min_fitness) if min_fitness else 0.0
    initial_val = min_fitness[0] if min_fitness else 0.0
    improvement_pct = (
        ((initial_val - best_val) / initial_val * 100) if initial_val > 0 else 0.0
    )

    fig.update_layout(
        title=dict(
            text=f"<span style='font-size: 13px; font-weight: 600; color: #0F172A;'>Curva de Convergencia do GA</span> <span style='font-size: 12px; color: #64748B;'>(Reducao de Custo: -{improvement_pct:.1f}%)</span>",
            x=0.02,
            y=0.96,
        ),
        xaxis=dict(
            title=dict(
                text="Geracoes",
                font=dict(family="Geist Sans, Inter, sans-serif", size=11, color="#64748B"),
            ),
            gridcolor="#F1F5F9",
            dtick=max(1, len(generations) // 8),
            tickfont=dict(family="SF Mono, JetBrains Mono, monospace", size=10, color="#64748B"),
        ),
        yaxis=dict(
            title=dict(
                text="Funcao de Fitness",
                font=dict(family="Geist Sans, Inter, sans-serif", size=11, color="#64748B"),
            ),
            gridcolor="#F1F5F9",
            tickfont=dict(family="SF Mono, JetBrains Mono, monospace", size=10, color="#64748B"),
        ),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FFFFFF",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            bgcolor="#FFFFFF",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(family="Geist Sans, Inter, sans-serif", size=11, color="#334155"),
        ),
        margin=dict(l=30, r=30, t=50, b=50),
    )

    return fig
