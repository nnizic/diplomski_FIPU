"""Eksperiment 1 - diplomski rad - Neven Nižić"""

import random

from deap import algorithms, base, creator, tools
import numpy as np
import pandas as pd

# ------------------------------
# Postavke
# ------------------------------
NUM_ACTIVITIES = 50
BUDGET = 1000
NUM_SIMULATIONS = 100  # Monte Carlo simulacija
POP_SIZE = 100  # Veličina populacije
NGEN = 40  # Generacije
CX_PB = 0.7  # Vjerojatnost križanja
MUT_PB = 0.2  # Vjerojatnost mutacije
SEED = 42
RUNS = 10  # Broj ponavljanja za svaku konfiguraciju

random.seed(SEED)
np.random.seed(SEED)


# ------------------------------
# Generiranje sintetičkih podataka
# ------------------------------
def generate_data():
    """Generira slučajne aktivnosti sa cijenom, trajanjem i ROI."""
    return [
        {
            "id": i,
            "cost": random.randint(50, 200),
            "optimistic": random.randint(5, 10),
            "realistic": random.randint(10, 20),
            "pessimistic": random.randint(20, 40),
            "roi": round(random.uniform(1.0, 3.0), 2),
        }
        for i in range(NUM_ACTIVITIES)
    ]


activities = generate_data()


# ------------------------------
# Monte Carlo evaluacija trajanja
# ------------------------------
def monte_carlo_eval_duration(individual):
    """Računa prosječno trajanje odabranih aktivnosti pomoću Monte Carlo simulacije."""
    selected = [act for i, act in enumerate(activities) if individual[i] == 1]
    if not selected:
        return 0.0

    durations = []
    for _ in range(NUM_SIMULATIONS):
        total = sum(
            random.triangular(act["optimistic"], act["realistic"], act["pessimistic"])
            for act in selected
        )
        durations.append(total)
    return np.mean(durations)


# ------------------------------
# Fitness funkcije
# ------------------------------
def calculate_metrics(individual):
    """Vraća ukupni trošak i ROI."""
    total_cost = sum(activities[i]["cost"] for i, sel in enumerate(individual) if sel)
    total_roi = sum(activities[i]["roi"] for i, sel in enumerate(individual) if sel)
    return total_cost, total_roi


def single_objective_fitness(individual):
    """Jedno-objektivni cilj: maksimizirati ROI, s kaznom za prekoračenje budžeta."""
    total_cost, total_roi = calculate_metrics(individual)
    if total_cost > BUDGET:
        return (-(total_cost - BUDGET),)
    return (total_roi,)


def multi_objective_fitness(individual):
    """Više-objektivni cilj: maksimizirati ROI, minimizirati trajanje."""
    total_cost, total_roi = calculate_metrics(individual)
    if total_cost > BUDGET:
        return 0, 99999
    avg_duration = monte_carlo_eval_duration(individual)
    return total_roi, avg_duration


# ------------------------------
# Kreiranje DEAP struktura
# ------------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("IndividualMulti", list, fitness=creator.FitnessMulti)


def create_toolbox(individual_type, evaluate_func, select_func, **kwargs):
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register(
        "individual",
        tools.initRepeat,
        individual_type,
        toolbox.attr_bool,
        NUM_ACTIVITIES,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("evaluate", evaluate_func)
    toolbox.register("select", select_func, **kwargs)
    return toolbox


toolbox_single = create_toolbox(
    creator.Individual, single_objective_fitness, tools.selTournament, tournsize=3
)
toolbox_multi = create_toolbox(
    creator.IndividualMulti, multi_objective_fitness, tools.selNSGA2
)


# ------------------------------
# Scenariji optimizacije
# ------------------------------
def run_random_search():
    """Random Search - traži najbolje rješenje slučajnim generiranjem."""
    best_ind, best_roi = None, -1
    num_evaluations = POP_SIZE * NGEN

    for _ in range(num_evaluations):
        ind = [random.randint(0, 1) for _ in range(NUM_ACTIVITIES)]
        cost, roi = calculate_metrics(ind)
        if cost <= BUDGET and roi > best_roi:
            best_ind, best_roi = ind, roi

    if best_ind is None:
        return 0, 0
    return best_roi, monte_carlo_eval_duration(best_ind)


def run_ga_single_objective():
    """GA optimizacija samo ROI-a."""
    pop = toolbox_single.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(
        pop,
        toolbox_single,
        cxpb=CX_PB,
        mutpb=MUT_PB,
        ngen=NGEN,
        halloffame=hof,
        verbose=False,
    )
    best = hof[0]
    return single_objective_fitness(best)[0], monte_carlo_eval_duration(best)


def run_ga_multi_objective():
    """Više-objektivni GA (NSGA-II)."""
    pop = toolbox_multi.population(n=POP_SIZE)
    hof = tools.ParetoFront()
    algorithms.eaMuPlusLambda(
        pop,
        toolbox_multi,
        mu=POP_SIZE,
        lambda_=POP_SIZE,
        cxpb=CX_PB,
        mutpb=MUT_PB,
        ngen=NGEN,
        halloffame=hof,
        verbose=False,
    )
    return list(hof)


# ------------------------------
# Ablation study
# ------------------------------
def run_ablation_study():
    """
    Ispituje utjecaj pojedinih komponenti algoritma:
    1. Standardni GA
    2. Bez mutacije
    3. Bez križanja
    4. Veći broj generacija
    5. Veća populacija
    """
    configs = [
        ("Standardni GA", CX_PB, MUT_PB, POP_SIZE, NGEN),
        ("Bez mutacije", CX_PB, 0.0, POP_SIZE, NGEN),
        ("Bez križanja", 0.0, MUT_PB, POP_SIZE, NGEN),
        ("Više generacija", CX_PB, MUT_PB, POP_SIZE, NGEN * 2),
        ("Veća populacija", CX_PB, MUT_PB, POP_SIZE * 2, NGEN),
    ]

    results = []

    for name, cxpb, mutpb, pop_size, ngen in configs:
        roi_scores = []
        dur_scores = []

        for _ in range(RUNS):
            pop = toolbox_single.population(n=pop_size)
            hof = tools.HallOfFame(1)
            algorithms.eaSimple(
                pop,
                toolbox_single,
                cxpb=cxpb,
                mutpb=mutpb,
                ngen=ngen,
                halloffame=hof,
                verbose=False,
            )
            best = hof[0]
            roi, dur = single_objective_fitness(best)[0], monte_carlo_eval_duration(
                best
            )
            roi_scores.append(roi)
            dur_scores.append(dur)

        results.append(
            {
                "Postavka": name,
                "ROI_mean": np.mean(roi_scores),
                "ROI_std": np.std(roi_scores),
                "Trajanje_mean": np.mean(dur_scores),
                "Trajanje_std": np.std(dur_scores),
            }
        )

    df = pd.DataFrame(results)
    df.to_csv("ablation_results.csv", index=False)
    print("\n--- Ablation Study Rezultati ---")
    print(df.to_string())
    print("Rezultati spremljeni u 'ablation_results.csv'")


# ------------------------------
# Pokretanje eksperimenata
# ------------------------------
def run_experiments():
    run_ablation_study()


if __name__ == "__main__":
    run_experiments()
