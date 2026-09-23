# Roteamento de Drones de Entrega com Estações de Recarga (GA + Streamlit)

Projeto acadêmico de Otimização e Algoritmos Genéticos aplicado à logística de entregas autônomas com restrições de bateria, recarga rápida e visualização interativa em 2D.

### Alunos:
- Luiz Gustavo Guaycurus Goulart
- Matheus Corrêa Cesar Biancovilli
- Nicholas Pompilio Coda

---

## Objetivo do Projeto
Resolver o problema de roteamento de um drone de entrega que parte de um depósito central, deve atender a um conjunto de clientes obrigatórios e retornar em segurança ao depósito, utilizando estações de recarga rápida intermediárias quando a sua autonomia energética for insuficiente.

---

## Estrutura do Repositório

```
.
├── requirements.txt         # streamlit, deap, plotly, numpy, pandas
├── src/
│   ├── __init__.py
│   ├── problem_model.py     # Classes para Nós, Depósito, Clientes, Estações de Recarga e Simulação
│   ├── ga_engine.py         # Algoritmo Genético (DEAP): cromossomo, operadores e fitness
│   └── visualizer.py        # Mapas interativos e gráficos de convergência com Plotly
├── tests/
│   └── test_drone_ga.py     # Bateria de testes automatizados e aferição de desempenho
├── app.py                   # Interface interativa Streamlit com controles na barra lateral
└── README.md
```

---

## Instalação e Execução Local

### 1. Criar e ativar o ambiente virtual (opcional, mas recomendado):
```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate
```

### 2. Instalar as dependências:
```bash
pip install -r requirements.txt
```

### 3. Rodar os testes de validação:
```bash
python tests/test_drone_ga.py
```

### 4. Iniciar a aplicação interativa:
```bash
streamlit run app.py
```

---

## Metodologia e Formulação Matemática

### Representação do Indivíduo (Cromossomo)
- **Cromossomo:** Permutação de índices inteiros $[c_1, c_2, ..., c_N]$ representando a ordem de visita aos clientes.
- **Inserção de Recarga:** Política inteligente de *Lookahead*. O simulador avalia se o drone alcança o próximo cliente e ainda possui energia residual para atingir o refúgio seguro mais próximo. Caso contrário, desvia para a estação de recarga com menor custo de desvio.

### Função de Aptidão (Fitness)
$$\min \quad \text{Fitness} = \sum_{(i,j) \in \text{Rota}} d_{ij} + (R \cdot P_{\text{recarga}}) + (K \cdot P_{\text{queda}})$$
- $d_{ij}$: Distância euclidiana percorrida.
- $R$: Número de recargas intermediárias realizadas.
- $P_{\text{recarga}}$: Custo de tempo/distância por recarga (ex.: 15 km eq.).
- $K$: Número de quedas registradas por bateria esgotada.
- $P_{\text{queda}}$: Penalidade severa por inviabilidade ($100.000$).

### Operadores Genéticos (DEAP)
- **Seleção:** Torneio de tamanho 3 (`selTournament`).
- **Crossover:** Order Crossover (`cxOrdered` - OX1).
- **Mutação:** Embaralhamento seletivo de índices (`mutShuffleIndexes`).
- **Elitismo:** *Hall of Fame* preservando as melhores soluções entre as gerações.
