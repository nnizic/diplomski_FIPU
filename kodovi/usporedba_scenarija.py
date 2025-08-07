"""Eksperimenti - Diplomski rad - Neven Nižić"""

import random

from deap import algorithms, base, creator, tools
import numpy as np
import pandas as pd

# ------------------------------
# Postavke
# ------------------------------
# Postavke problema
NUM_ACTIVITIES = 10
BUDGET = 1000
NUM_SIMULATIONS = 100  # Monte Carlo simulacija

# Postavke algoritma
POP_SIZE = 200
NGEN = 40
CX_PB = 0.7
MUT_PB = 0.2

# Postavke eksperimenta
RUNS = 10  # Broj ponavljanja svakog scenarija za statističku značajnost
SEED = 42

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

    durations = [
        sum(
            random.triangular(act["optimistic"], act["realistic"], act["pessimistic"])
            for act in selected
        )
        for _ in range(NUM_SIMULATIONS)
    ]
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
# Funkcije za JEDNO pokretanje scenarija
# ------------------------------
def run_random_search_once():
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


def run_ga_single_objective_once():
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


def run_ga_multi_objective_once():
    """Više-objektivni GA (NSGA-II) - vraća reprezentativno rješenje."""
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

    if not hof:
        return 0, 0  # Nije pronađeno nijedno rješenje

    # Iz Paretovog fronta biramo jedno reprezentativno rješenje za statistiku
    # Dobar izbor je ono s najvišim ROI-em
    best_solution = max(hof, key=lambda ind: ind.fitness.values[0])
    return best_solution.fitness.values


# ------------------------------
# Glavni program za usporedbu scenarija
# ------------------------------
def run_comparative_study():
    """
    Provodi statističku usporedbu tri glavna scenarija:
    1. Random Search (MC)
    2. GA (samo ROI)
    3. GA + MC (NSGA-II)
    Svaki scenarij se pokreće 'RUNS' puta za pouzdane rezultate.
    """
    scenarios = {
        "Random Search (MC)": run_random_search_once,
        "GA (samo ROI)": run_ga_single_objective_once,
        "GA+MC (NSGA-II)": run_ga_multi_objective_once,
    }

    all_results = []

    for name, run_function in scenarios.items():
        print(f"--- Pokrećem scenarij: {name} ({RUNS} puta) ---")

        # Liste za prikupljanje rezultata iz svih pokretanja
        run_rois = []
        run_durations = []

        for i in range(RUNS):
            # Pokreni jednu instancu scenarija
            roi, duration = run_function()
            run_rois.append(roi)
            run_durations.append(duration)
            print(f"  Run {i+1}/{RUNS}: ROI={roi:.2f}, Trajanje={duration:.2f}")

        # Izračunaj statistiku za ovaj scenarij
        all_results.append(
            {
                "Scenarij": name,
                "ROI_mean": np.mean(run_rois),
                "ROI_std": np.std(run_rois),
                "Trajanje_mean": np.mean(run_durations),
                "Trajanje_std": np.std(run_durations),
            }
        )
        print("-" * 50)

    # Kreiraj i spremi konačni DataFrame
    df = pd.DataFrame(all_results)
    df.to_csv("usporedni_rezultati.csv", index=False)

    print("\n--- Konačni usporedni rezultati (statistika preko 10 pokretanja) ---")
    print(df.to_string())
    print("\n✅ Rezultati spremljeni u 'usporedni_rezultati.csv'")


if __name__ == "__main__":
    run_comparative_study()
