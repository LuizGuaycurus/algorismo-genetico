"""
ga_engine.py
============
Mecanismo de Algoritmo Genético construído com a biblioteca DEAP para
otimização do Roteamento de Drones com Estações de Recarga.
"""

import random
import time
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from deap import base, creator, tools

from src.problem_model import DroneProblem, RouteResult


def setup_deap_environment():
    """
    Inicializa os tipos do DEAP de forma segura, evitando recriações duplicadas.
    """
    if not hasattr(creator, "FitnessMin"):
        # Minimizar fitness (distância + penalidades)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    if not hasattr(creator, "Individual"):
        # Indivíduo representado como permutação de índices de clientes
        creator.create("Individual", list, fitness=creator.FitnessMin)


class GeneticAlgorithmSolver:
    """
    Controlador do Algoritmo Genético com DEAP para resolução do Drone Routing.
    """

    def __init__(
        self,
        problem: DroneProblem,
        population_size: int = 50,
        generations: int = 50,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.1,
        tournament_size: int = 3,
        elitism_size: int = 2,
        seed: Optional[int] = None,
    ):
        self.problem = problem
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.crossover_prob = float(crossover_prob)
        self.mutation_prob = float(mutation_prob)
        self.tournament_size = int(tournament_size)
        self.elitism_size = max(1, int(elitism_size))
        self.seed = seed

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        setup_deap_environment()
        self.toolbox = base.Toolbox()
        self._setup_toolbox()

    def _setup_toolbox(self):
        """Configura operadores genéticos e funções auxiliares no toolbox do DEAP."""
        num_customers = len(self.problem.customers)

        # Gerador de cromossomo: permutação aleatória de índices [0, 1, ..., N-1]
        self.toolbox.register(
            "indices", random.sample, range(num_customers), num_customers
        )
        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, self.toolbox.indices
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        # Função de avaliação (Fitness)
        self.toolbox.register("evaluate", self._evaluate_individual)

        # Operadores Genéticos exigidos
        # 1. Cruzamento para permutações: Order Crossover (OX1)
        self.toolbox.register("mate", tools.cxOrdered)

        # 2. Mutação por troca / inversão de índices
        self.toolbox.register(
            "mutate", tools.mutShuffleIndexes, indpb=2.0 / max(1, num_customers)
        )

        # 3. Seleção por Torneio
        self.toolbox.register(
            "select", tools.selTournament, tournsize=self.tournament_size
        )

    def _evaluate_individual(self, individual: List[int]) -> Tuple[float]:
        """
        Calcula o fitness do indivíduo simulando a viagem do drone com recargas.
        Retorna tupla com (fitness,) conforme convenção do DEAP.
        """
        route_result = self.problem.evaluate_route(individual)
        return (route_result.fitness,)

    def solve(self) -> Tuple[List[int], RouteResult, Dict[str, List[Any]], float]:
        """
        Executa a evolução do Algoritmo Genético com elitismo.

        Retorna:
            best_individual: Melhor permutação de clientes encontrada.
            best_route_result: Objeto RouteResult detalhado com pernas e consumo.
            history: Histórico de convergência contendo gerações, min_fitness e avg_fitness.
            elapsed_time: Tempo de execução em segundos.
        """
        start_time = time.perf_counter()

        # Criação da população inicial
        pop = self.toolbox.population(n=self.population_size)

        # Hall of Fame para garantir elitismo
        hof = tools.HallOfFame(self.elitism_size)

        # Estatísticas para monitoramento de convergência
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("min", np.min)
        stats.register("avg", np.mean)

        history = {
            "generation": [],
            "min_fitness": [],
            "avg_fitness": [],
        }

        # Avaliação da população inicial
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        hof.update(pop)
        record = stats.compile(pop)
        history["generation"].append(0)
        history["min_fitness"].append(float(record["min"]))
        history["avg_fitness"].append(float(record["avg"]))

        # Ciclo geracional
        for gen in range(1, self.generations + 1):
            # Seleção com reserva para elitismo
            offspring = self.toolbox.select(pop, len(pop) - self.elitism_size)
            offspring = list(map(self.toolbox.clone, offspring))

            # Aplicação do Crossover (Order Crossover)
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Aplicação da Mutação
            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Avaliação dos indivíduos modificados
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Elitismo: os melhores do Hall of Fame são preservados diretamente
            elite = [self.toolbox.clone(ind) for ind in hof]
            pop[:] = offspring + elite

            # Atualização do Hall of Fame com a nova população
            hof.update(pop)

            # Registro das métricas da geração
            record = stats.compile(pop)
            history["generation"].append(gen)
            history["min_fitness"].append(float(record["min"]))
            history["avg_fitness"].append(float(record["avg"]))

        elapsed_time = time.perf_counter() - start_time

        best_individual = list(hof[0])
        best_route_result = self.problem.evaluate_route(best_individual)

        return best_individual, best_route_result, history, elapsed_time
