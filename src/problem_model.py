"""
problem_model.py
================
Modelagem matemática e estruturas de dados para o problema de
Roteamento de Drones de Entrega com Estações de Recarga.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math
import numpy as np


@dataclass
class Node:
    """Classe base para nós no plano cartesiano 2D."""
    id: int
    name: str
    x: float
    y: float
    node_type: str  # 'depot', 'customer', 'recharge_station'

    def distance_to(self, other: "Node") -> float:
        """Calcula a distância euclidiana até outro nó."""
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class Depot(Node):
    """Ponto de partida e retorno do drone."""
    node_type: str = "depot"


@dataclass
class Customer(Node):
    """Ponto de entrega obrigatório."""
    demand: float = 1.0
    node_type: str = "customer"


@dataclass
class RechargeStation(Node):
    """Estação de recarga rápida onde a bateria é restaurada para 100%."""
    node_type: str = "recharge_station"


@dataclass
class RouteLeg:
    """Representa um trecho do trajeto percorrido pelo drone."""
    leg_index: int
    from_node: Node
    to_node: Node
    distance: float
    battery_before: float
    battery_after: float
    battery_pct_after: float
    is_recharge: bool = False
    crashed: bool = False
    event_description: str = ""


@dataclass
class RouteResult:
    """Resultado detalhado da simulação de uma rota avaliada."""
    order: List[int]
    full_path: List[Node]
    legs: List[RouteLeg]
    total_distance: float
    recharge_count: int
    recharge_stations_used: List[int]
    is_feasible: bool
    crashes: int
    fitness: float


class DroneProblem:
    """
    Representa uma instância do problema de roteamento de drone com recarga.
    """

    def __init__(
        self,
        depot: Depot,
        customers: List[Customer],
        recharge_stations: List[RechargeStation],
        battery_capacity: float = 25.0,
        consumption_rate: float = 1.0,
        recharge_penalty_time: float = 15.0,
        battery_penalty: float = 100000.0,
    ):
        self.depot = depot
        self.customers = customers
        self.recharge_stations = recharge_stations
        self.battery_capacity = float(battery_capacity)
        self.consumption_rate = float(consumption_rate)
        self.recharge_penalty_time = float(recharge_penalty_time)
        self.battery_penalty = float(battery_penalty)

        # Mapeamento rápido de refúgios (depósito + estações de recarga)
        self.refuges: List[Node] = [self.depot] + self.recharge_stations

    @classmethod
    def generate_random_scenario(
        cls,
        num_customers: int = 12,
        num_recharge_stations: int = 2,
        area_size: float = 30.0,
        battery_capacity: float = 25.0,
        seed: Optional[int] = 42,
    ) -> "DroneProblem":
        """
        Gera um cenário sintético aleatório e reprodutível.
        Depósito centralizado em (0, 0), com clientes e estações distribuídos no espaço.
        """
        rng = np.random.default_rng(seed)

        depot = Depot(id=0, name="Depósito Central", x=0.0, y=0.0)

        # Clientes distribuídos em torno do depósito
        customers: List[Customer] = []
        for i in range(num_customers):
            # Coordenadas em [-area_size/2, area_size/2]
            cx = float(rng.uniform(-area_size / 2, area_size / 2))
            cy = float(rng.uniform(-area_size / 2, area_size / 2))
            customers.append(
                Customer(id=i + 1, name=f"Cliente {i + 1}", x=round(cx, 2), y=round(cy, 2))
            )

        # Estações de recarga estrategicamente posicionadas nos quadrantes
        stations: List[RechargeStation] = []
        for j in range(num_recharge_stations):
            # Posicionamento angular para cobrir a área
            angle = (2 * math.pi * j / num_recharge_stations) + float(rng.uniform(-0.3, 0.3))
            radius = float(rng.uniform(area_size * 0.25, area_size * 0.45))
            sx = float(radius * math.cos(angle))
            sy = float(radius * math.sin(angle))
            stations.append(
                RechargeStation(
                    id=num_customers + j + 1,
                    name=f"Estação {j + 1}",
                    x=round(sx, 2),
                    y=round(sy, 2),
                )
            )

        return cls(
            depot=depot,
            customers=customers,
            recharge_stations=stations,
            battery_capacity=battery_capacity,
        )

    def min_safe_distance_to_refuge(self, node: Node) -> float:
        """Calcula a menor distância euclidiana do nó até o refúgio mais próximo."""
        return min(node.distance_to(refuge) for refuge in self.refuges)

    def find_best_recharge_station(
        self, current_node: Node, target_node: Node, current_battery: float
    ) -> Optional[RechargeStation]:
        """
        Busca a estação de recarga alcançável com a bateria atual que minimize
        o desvio total até o nó de destino.
        """
        best_station = None
        best_cost = float("inf")

        for station in self.recharge_stations:
            dist_to_station = current_node.distance_to(station)
            needed_batt = dist_to_station * self.consumption_rate
            if needed_batt <= current_battery:
                # Custo da rota passando pela estação: dist(atual -> estação) + dist(estação -> destino)
                cost = dist_to_station + station.distance_to(target_node)
                if cost < best_cost:
                    best_cost = cost
                    best_station = station

        return best_station

    def evaluate_route(self, customer_order: List[int]) -> RouteResult:
        """
        Simula a viagem do drone seguindo a permutação de clientes especificada.
        Aplica lookahead com inserção inteligente de recargas para garantir voo seguro.
        """
        current_node: Node = self.depot
        current_battery = self.battery_capacity
        total_distance = 0.0
        recharge_count = 0
        crashes = 0
        legs: List[RouteLeg] = []
        full_path: List[Node] = [self.depot]
        stations_used: List[int] = []
        leg_index = 1

        for cust_idx in customer_order:
            target = self.customers[cust_idx]
            dist_to_target = current_node.distance_to(target)
            batt_to_target = dist_to_target * self.consumption_rate

            # Verificação de segurança (Lookahead):
            # O drone consegue alcançar o alvo E de lá alcançar um refúgio seguro?
            min_refuge_dist = self.min_safe_distance_to_refuge(target)
            batt_to_safe_refuge = min_refuge_dist * self.consumption_rate

            can_visit_safely = (
                current_battery >= batt_to_target
                and (current_battery - batt_to_target) >= batt_to_safe_refuge
            )

            if can_visit_safely:
                # Voo direto até o cliente
                batt_after = current_battery - batt_to_target
                legs.append(
                    RouteLeg(
                        leg_index=leg_index,
                        from_node=current_node,
                        to_node=target,
                        distance=dist_to_target,
                        battery_before=current_battery,
                        battery_after=batt_after,
                        battery_pct_after=(batt_after / self.battery_capacity) * 100,
                        is_recharge=False,
                        crashed=False,
                        event_description=f"Entrega no {target.name}",
                    )
                )
                leg_index += 1
                total_distance += dist_to_target
                current_battery = batt_after
                current_node = target
                full_path.append(target)
            else:
                # Necessidade de recarga antes de alcançar o cliente
                station = self.find_best_recharge_station(current_node, target, current_battery)
                if station is not None:
                    # 1. Voo até a estação de recarga
                    dist_to_st = current_node.distance_to(station)
                    batt_to_st = dist_to_st * self.consumption_rate
                    batt_after_st = max(0.0, current_battery - batt_to_st)
                    legs.append(
                        RouteLeg(
                            leg_index=leg_index,
                            from_node=current_node,
                            to_node=station,
                            distance=dist_to_st,
                            battery_before=current_battery,
                            battery_after=self.battery_capacity,  # Restaura a 100%
                            battery_pct_after=100.0,
                            is_recharge=True,
                            crashed=False,
                            event_description=f"Recarga em {station.name} (Bateria restaurada para 100%)",
                        )
                    )
                    leg_index += 1
                    total_distance += dist_to_st
                    recharge_count += 1
                    stations_used.append(station.id)
                    current_battery = self.battery_capacity
                    current_node = station
                    full_path.append(station)

                    # 2. Voo da estação até o cliente
                    dist_st_to_target = station.distance_to(target)
                    batt_to_target_from_st = dist_st_to_target * self.consumption_rate
                    if batt_to_target_from_st <= current_battery:
                        batt_after = current_battery - batt_to_target_from_st
                        legs.append(
                            RouteLeg(
                                leg_index=leg_index,
                                from_node=station,
                                to_node=target,
                                distance=dist_st_to_target,
                                battery_before=current_battery,
                                battery_after=batt_after,
                                battery_pct_after=(batt_after / self.battery_capacity) * 100,
                                is_recharge=False,
                                crashed=False,
                                event_description=f"Entrega no {target.name}",
                            )
                        )
                        leg_index += 1
                        total_distance += dist_st_to_target
                        current_battery = batt_after
                        current_node = target
                        full_path.append(target)
                    else:
                        # Queda: cliente inalcançável mesmo saindo com 100% de bateria
                        legs.append(
                            RouteLeg(
                                leg_index=leg_index,
                                from_node=station,
                                to_node=target,
                                distance=dist_st_to_target,
                                battery_before=current_battery,
                                battery_after=0.0,
                                battery_pct_after=0.0,
                                is_recharge=False,
                                crashed=True,
                                event_description=f"Queda: bateria esgotada ao tentar alcancar {target.name}",
                            )
                        )
                        leg_index += 1
                        total_distance += dist_st_to_target
                        crashes += 1
                        current_battery = 0.0
                        current_node = target
                        full_path.append(target)
                else:
                    # Nenhuma estação alcançável: drone tenta ir até o cliente e cai
                    legs.append(
                        RouteLeg(
                            leg_index=leg_index,
                            from_node=current_node,
                            to_node=target,
                            distance=dist_to_target,
                            battery_before=current_battery,
                            battery_after=0.0,
                            battery_pct_after=0.0,
                            is_recharge=False,
                            crashed=True,
                            event_description=f"Queda: drone sem autonomia para alcancar {target.name} ou estacao",
                        )
                    )
                    leg_index += 1
                    total_distance += dist_to_target
                    crashes += 1
                    current_battery = 0.0
                    current_node = target
                    full_path.append(target)

        # Retorno final ao Depósito Central
        dist_to_depot = current_node.distance_to(self.depot)
        batt_to_depot = dist_to_depot * self.consumption_rate

        if batt_to_depot <= current_battery:
            batt_after = current_battery - batt_to_depot
            legs.append(
                RouteLeg(
                    leg_index=leg_index,
                    from_node=current_node,
                    to_node=self.depot,
                    distance=dist_to_depot,
                    battery_before=current_battery,
                    battery_after=batt_after,
                    battery_pct_after=(batt_after / self.battery_capacity) * 100,
                    is_recharge=False,
                    crashed=False,
                    event_description=f"Pouso seguro de retorno ao {self.depot.name}",
                )
            )
            total_distance += dist_to_depot
            current_battery = batt_after
            full_path.append(self.depot)
        else:
            # Tenta desvio para estação antes de retornar ao depósito
            station = self.find_best_recharge_station(current_node, self.depot, current_battery)
            if station is not None:
                # Voo para estação
                dist_to_st = current_node.distance_to(station)
                legs.append(
                    RouteLeg(
                        leg_index=leg_index,
                        from_node=current_node,
                        to_node=station,
                        distance=dist_to_st,
                        battery_before=current_battery,
                        battery_after=self.battery_capacity,
                        battery_pct_after=100.0,
                        is_recharge=True,
                        crashed=False,
                        event_description=f"Recarga em {station.name} para retorno seguro",
                    )
                )
                leg_index += 1
                total_distance += dist_to_st
                recharge_count += 1
                stations_used.append(station.id)
                current_battery = self.battery_capacity
                current_node = station
                full_path.append(station)

                # Estação até Depósito
                dist_st_depot = station.distance_to(self.depot)
                batt_st_depot = dist_st_depot * self.consumption_rate
                if batt_st_depot <= current_battery:
                    batt_after = current_battery - batt_st_depot
                    legs.append(
                        RouteLeg(
                            leg_index=leg_index,
                            from_node=station,
                            to_node=self.depot,
                            distance=dist_st_depot,
                            battery_before=current_battery,
                            battery_after=batt_after,
                            battery_pct_after=(batt_after / self.battery_capacity) * 100,
                            is_recharge=False,
                            crashed=False,
                            event_description=f"Pouso seguro de retorno ao {self.depot.name}",
                        )
                    )
                    total_distance += dist_st_depot
                    current_battery = batt_after
                    full_path.append(self.depot)
                else:
                    legs.append(
                        RouteLeg(
                            leg_index=leg_index,
                            from_node=station,
                            to_node=self.depot,
                            distance=dist_st_depot,
                            battery_before=current_battery,
                            battery_after=0.0,
                            battery_pct_after=0.0,
                            is_recharge=False,
                            crashed=True,
                            event_description=f"Queda: bateria esgotada no retorno ao {self.depot.name}",
                        )
                    )
                    total_distance += dist_st_depot
                    crashes += 1
                    current_battery = 0.0
                    full_path.append(self.depot)
            else:
                legs.append(
                    RouteLeg(
                        leg_index=leg_index,
                        from_node=current_node,
                        to_node=self.depot,
                        distance=dist_to_depot,
                        battery_before=current_battery,
                        battery_after=0.0,
                        battery_pct_after=0.0,
                        is_recharge=False,
                        crashed=True,
                        event_description=f"Queda: drone sem bateria no retorno ao {self.depot.name}",
                    )
                )
                total_distance += dist_to_depot
                crashes += 1
                current_battery = 0.0
                full_path.append(self.depot)

        is_feasible = (crashes == 0)

        # Função de Custo / Fitness a ser minimizada:
        # Distância + (Recargas * Penalidade de Tempo) + (Quedas * Penalidade Severa)
        fitness = (
            total_distance
            + (recharge_count * self.recharge_penalty_time)
            + (crashes * self.battery_penalty)
        )

        return RouteResult(
            order=customer_order,
            full_path=full_path,
            legs=legs,
            total_distance=round(total_distance, 2),
            recharge_count=recharge_count,
            recharge_stations_used=stations_used,
            is_feasible=is_feasible,
            crashes=crashes,
            fitness=round(fitness, 2),
        )
