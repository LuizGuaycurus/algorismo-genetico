"""
test_drone_ga.py
================
Suite de testes automatizados para validação do modelo de drone,
algoritmo genético (DEAP) e visualizações Plotly.
"""

import sys
import os
import time

# Garante que a raiz do projeto esteja no sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.problem_model import DroneProblem, Depot, Customer, RechargeStation
from src.ga_engine import GeneticAlgorithmSolver
from src.visualizer import plot_route, plot_convergence


def test_scenario_generation_and_evaluation():
    print("-> Testando geração de cenário e simulação de rota...")
    problem = DroneProblem.generate_random_scenario(
        num_customers=10,
        num_recharge_stations=2,
        battery_capacity=30.0,
        seed=123,
    )

    assert len(problem.customers) == 10
    assert len(problem.recharge_stations) == 2
    assert problem.depot.x == 0.0 and problem.depot.y == 0.0

    # Rota simples sequencial [0, 1, ..., 9]
    order = list(range(10))
    result = problem.evaluate_route(order)

    assert result.total_distance > 0
    assert result.full_path[0] == problem.depot
    assert result.full_path[-1] == problem.depot
    assert len(result.legs) >= 11  # No mínimo 10 clientes + retorno ao depósito

    # Verifica integridade das pernas
    for leg in result.legs:
        assert leg.distance >= 0
        if not leg.crashed:
            assert leg.battery_after >= 0

    print("  [OK] Simulação de rota e integridade de nós validadas.")


def test_crash_penalty_on_low_battery():
    print("-> Testando detecção de queda e aplicação de penalidade...")
    # Bateria minúscula (2 km) em área de 30 km -> inevitável queda
    problem = DroneProblem.generate_random_scenario(
        num_customers=10,
        num_recharge_stations=1,
        battery_capacity=2.0,
        seed=42,
    )
    order = list(range(10))
    result = problem.evaluate_route(order)

    assert not result.is_feasible, "A rota deveria ser inviável com bateria de 2 km."
    assert result.crashes > 0, "Deveria registrar quedas por falta de bateria."
    assert result.fitness >= problem.battery_penalty, "Fitness deve conter penalidade severa."
    print("  [OK] Penalidade severa por queda ativada com sucesso.")


def test_ga_performance_and_convergence():
    print("-> Testando execução do GA (DEAP) e métrica de desempenho (< 2 segundos)...")
    problem = DroneProblem.generate_random_scenario(
        num_customers=12,
        num_recharge_stations=2,
        battery_capacity=25.0,
        seed=42,
    )

    solver = GeneticAlgorithmSolver(
        problem=problem,
        population_size=50,
        generations=50,
        crossover_prob=0.8,
        mutation_prob=0.1,
        tournament_size=3,
        elitism_size=2,
        seed=42,
    )

    start = time.perf_counter()
    best_ind, best_route, history, elapsed = solver.solve()
    total_time = time.perf_counter() - start

    print(f"  Tempo de execução: {total_time:.3f} s (GA interno: {elapsed:.3f} s)")
    assert total_time < 2.0, f"O GA demorou mais que 2 segundos ({total_time:.3f}s)!"

    # Validação do cromossomo
    assert len(best_ind) == 12, "O cromossomo deve conter exatamente 12 clientes."
    assert sorted(best_ind) == list(range(12)), "O cromossomo deve ser uma permutação perfeita sem duplicatas."

    # Validação da convergência
    assert len(history["min_fitness"]) == 51  # Geração 0 até 50
    assert history["min_fitness"][-1] <= history["min_fitness"][0], "O fitness deve melhorar ou manter o valor inicial."

    print(f"  Fitness inicial: {history['min_fitness'][0]:.2f} -> Fitness final: {history['min_fitness'][-1]:.2f}")
    print("  [OK] Desempenho excelente e convergência validada.")


def test_visualizer():
    print("-> Testando geração de figuras Plotly...")
    problem = DroneProblem.generate_random_scenario(
        num_customers=8,
        num_recharge_stations=2,
        battery_capacity=25.0,
        seed=99,
    )
    result = problem.evaluate_route(list(range(8)))

    fig_map = plot_route(problem, result)
    assert fig_map is not None
    assert len(fig_map.data) > 0

    history = {
        "generation": [0, 1, 2],
        "min_fitness": [100.0, 90.0, 80.0],
        "avg_fitness": [120.0, 110.0, 95.0],
    }
    fig_conv = plot_convergence(history)
    assert fig_conv is not None
    assert len(fig_conv.data) == 2

    print("  [OK] Gráficos Plotly gerados com sucesso.")


if __name__ == "__main__":
    test_scenario_generation_and_evaluation()
    test_crash_penalty_on_low_battery()
    test_ga_performance_and_convergence()
    test_visualizer()
    print("\n[SUCESSO] TODOS OS TESTES PASSARAM COM EXCELENCIA!")
