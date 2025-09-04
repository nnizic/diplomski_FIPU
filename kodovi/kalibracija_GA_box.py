"""Eksperiment 1 i vizualizacija kutijasti graf - diplomski rad - Neven Nižić"""

import random
import time

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# KONFIGURACIJA (za Eksperiment 1)
# ==============================================================================
CONFIG = {
    "NUM_ACTIVITIES": 50,
    "BUDGET": 1000,
    "NUM_SIMULATIONS": 100,
    "SEED": 42,
    "RUNS": 10,  # Broj ponavljanja za svaku konfiguraciju
}

# Inicijalni GA parametri (Standardni GA)
POP_SIZE = 100
NGEN = 40
CX_PB = 0.7
MUT_PB = 0.2

random.seed(CONFIG["SEED"])
np.random.seed(CONFIG["SEED"])

# ==============================================================================
# DEAP STRUKTURE
# ==============================================================================
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


# ==============================================================================
# POMOĆNE FUNKCIJE
# ==============================================================================
def generate_data(num_activities):
    """Generira slučajne aktivnosti."""
    return [
        {
            "id": i,
            "cost": random.randint(50, 200),
            "optimistic": random.randint(5, 10),
            "realistic": random.randint(10, 20),
            "pessimistic": random.randint(20, 40),
            "roi": round(random.uniform(1.0, 3.0), 2),
        }
        for i in range(num_activities)
    ]


activities = generate_data(CONFIG["NUM_ACTIVITIES"])


def monte_carlo_eval_duration(individual):
    """Računa prosječno trajanje pomoću Monte Carlo simulacije."""
    selected = [act for i, act in enumerate(activities) if individual[i] == 1]
    if not selected:
        return 0.0

    durations = [
        sum(
            random.triangular(act["optimistic"], act["realistic"], act["pessimistic"])
            for act in selected
        )
        for _ in range(CONFIG["NUM_SIMULATIONS"])
    ]
    return np.mean(durations)


def calculate_metrics(individual):
    """Vraća ukupni trošak i ROI."""
    total_cost = sum(activities[i]["cost"] for i, sel in enumerate(individual) if sel)
    total_roi = sum(activities[i]["roi"] for i, sel in enumerate(individual) if sel)
    return total_cost, total_roi


def single_objective_fitness(individual):
    """Jedno-kriterijski fitness."""
    total_cost, total_roi = calculate_metrics(individual)
    if total_cost > CONFIG["BUDGET"]:
        return (-(total_cost - CONFIG["BUDGET"]),)
    return (total_roi,)


# ==============================================================================
# EKSPERIMENT 1: ABLACIJSKA STUDIJA
# ==============================================================================
def run_ablation_study():
    """Ispituje utjecaj pojedinih komponenti i sprema sirove rezultate."""
    print("\n===== ZAPOČINJEM EKSPERIMENT 1: Ablacijska studija =====")

    # Lista konfiguracija koje ćemo testirati
    configs_ablation = [
        (
            "Standardni GA",
            {"cxpb": CX_PB, "mutpb": MUT_PB, "pop_size": POP_SIZE, "ngen": NGEN},
        ),
        (
            "Bez mutacije",
            {"cxpb": CX_PB, "mutpb": 0.0, "pop_size": POP_SIZE, "ngen": NGEN},
        ),
        (
            "Bez križanja",
            {"cxpb": 0.0, "mutpb": MUT_PB, "pop_size": POP_SIZE, "ngen": NGEN},
        ),
        (
            "Više generacija",
            {"cxpb": CX_PB, "mutpb": MUT_PB, "pop_size": POP_SIZE, "ngen": NGEN * 2},
        ),
        (
            "Veća populacija",
            {"cxpb": CX_PB, "mutpb": MUT_PB, "pop_size": POP_SIZE * 2, "ngen": NGEN},
        ),
    ]

    # IZMJENA: Spremanje sirovih rezultata umjesto agregiranih
    raw_results = []

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_bool,
        CONFIG["NUM_ACTIVITIES"],
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("evaluate", single_objective_fitness)
    toolbox.register("select", tools.selTournament, tournsize=3)

    for name, params in configs_ablation:
        print(f"--- Pokrećem konfiguraciju: {name} ({CONFIG['RUNS']} puta) ---")

        for i in range(CONFIG["RUNS"]):
            pop = toolbox.population(n=params["pop_size"])
            hof = tools.HallOfFame(1)

            algorithms.eaSimple(
                pop,
                toolbox,
                cxpb=params["cxpb"],
                mutpb=params["mutpb"],
                ngen=params["ngen"],
                halloffame=hof,
                verbose=False,
            )

            best_ind = hof[0]

            final_roi = single_objective_fitness(best_ind)[0]
            final_duration = monte_carlo_eval_duration(best_ind)

            # Dodajemo sirovi rezultat u listu
            raw_results.append(
                {
                    "Konfiguracija": name,
                    "Run": i + 1,
                    "ROI": final_roi,
                    "Trajanje": final_duration,
                }
            )
            print(f"  Run {i+1}: ROI={final_roi:.3f}, Trajanje={final_duration:.3f}")

    # Spremanje sirovih podataka u CSV
    df_raw = pd.DataFrame(raw_results)
    df_raw.to_csv("ablation_study_raw_results.csv", index=False)
    print(
        "\nSirovi rezultati ablacijske studije spremljeni u 'ablation_study_raw_results.csv'"
    )


# ==============================================================================
# VIZUALIZACIJA (Kutijasti graf)
# ==============================================================================
def plot_ablation_boxplot():
    """Učitava sirove rezultate ablacijske studije i stvara kutijaste grafe."""
    try:
        df = pd.read_csv("ablation_study_raw_results.csv")
    except FileNotFoundError:
        print("Datoteka 'ablation_study_raw_results.csv' nije pronađena.")
        return

    order = [
        "Bez križanja",
        "Bez mutacije",
        "Standardni GA",
        "Više generacija",
        "Veća populacija",
    ]
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Usporedba performansi i stabilnosti GA konfiguracija", fontsize=18)

    sns.boxplot(
        ax=axes[0], x="Konfiguracija", y="ROI", data=df, order=order, palette="viridis"
    )
    axes[0].set_title("Distribucija ROI vrijednosti", fontsize=14)
    axes[0].set_xlabel("Konfiguracija GA", fontsize=12)
    axes[0].set_ylabel("ROI", fontsize=12)
    axes[0].tick_params(axis="x", rotation=45)

    sns.boxplot(
        ax=axes[1],
        x="Konfiguracija",
        y="Trajanje",
        data=df,
        order=order,
        palette="plasma",
    )
    axes[1].set_title("Distribucija procijenjenog trajanja", fontsize=14)
    axes[1].set_xlabel("Konfiguracija GA", fontsize=12)
    axes[1].set_ylabel("Prosječno trajanje (dani)", fontsize=12)
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Kreiranje foldera ako ne postoji
    import os

    if not os.path.exists("graf_box"):
        os.makedirs("graf_box")

    output_filename = "graf_box/ga_usporedba_boxplot.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Kutijasti graf spremljen u '{output_filename}'")


# ==============================================================================
# GLAVNI POKRETAČ
# ==============================================================================
if __name__ == "__main__":
    start_time = time.time()

    # 1. Provedi ablacijsku studiju (Eksperiment 1)
    run_ablation_study()

    # 2. Generiraj kutijasti graf za rezultate
    plot_ablation_boxplot()

    end_time = time.time()
    print(
        f"\nUkupno vrijeme izvođenja Eksperimenta 1: {end_time - start_time:.2f} sekundi."
    )
