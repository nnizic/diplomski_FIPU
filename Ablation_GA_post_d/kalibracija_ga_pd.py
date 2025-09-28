"""Eksperiment 1 i vizualizacija kutijasti graf za više scenarija"""

import random
import time
import os

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# KONFIGURACIJA
# ==============================================================================
CONFIG = {
    "SEED": 42,
    "RUNS": 10,  # Broj ponavljanja
    "NUM_SIMULATIONS": 100,  # Monte Carlo iteracije
    "experimental_series": [
        {"name": "A1_Osnovni", "NUM_ACTIVITIES": 10, "BUDGET": 1000},
        {"name": "A2_Srednji", "NUM_ACTIVITIES": 50, "BUDGET": 2500},
        {"name": "A3_Slozeni", "NUM_ACTIVITIES": 100, "BUDGET": 5000},
        {"name": "B1_Restriktivan", "NUM_ACTIVITIES": 50, "BUDGET": 1500},
        {"name": "B3_Labav", "NUM_ACTIVITIES": 50, "BUDGET": 4000},
    ],
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
    """Generira slučajne aktivnosti s validnim trokutastim parametrima."""
    data = []
    for i in range(num_activities):
        optimistic = random.randint(5, 10)
        pessimistic = random.randint(20, 40)
        realistic = random.randint(optimistic, pessimistic)  # osigurava low <= mode <= high
        roi = round(random.uniform(1.0, 3.0), 2)
        cost = random.randint(50, 200)
        data.append({
            "id": i,
            "cost": cost,
            "optimistic": optimistic,
            "realistic": realistic,
            "pessimistic": pessimistic,
            "roi": roi,
        })
    return data

# ==============================================================================
# EKSPERIMENT: Ablacijska studija za više scenarija
# ==============================================================================
def run_ablation_study():
    """Prolazi kroz sve scenarije i GA konfiguracije, sprema sirove rezultate."""
    print("\n===== ZAPOČINJEM EKSPERIMENT: Ablacijska studija =====")
    
    configs_ablation = [
        ("Standardni GA", {"cxpb": CX_PB, "mutpb": MUT_PB, "pop_size": POP_SIZE, "ngen": NGEN}),
        ("Bez mutacije", {"cxpb": CX_PB, "mutpb": 0.0, "pop_size": POP_SIZE, "ngen": NGEN}),
        ("Bez križanja", {"cxpb": 0.0, "mutpb": MUT_PB, "pop_size": POP_SIZE, "ngen": NGEN}),
        ("Više generacija", {"cxpb": CX_PB, "mutpb": MUT_PB, "pop_size": POP_SIZE, "ngen": NGEN * 2}),
        ("Veća populacija", {"cxpb": CX_PB, "mutpb": MUT_PB, "pop_size": POP_SIZE * 2, "ngen": NGEN}),
    ]

    all_results = []

    for scenario in CONFIG["experimental_series"]:
        print(f"\n>>> Pokrećem scenarij: {scenario['name']} | "
              f"Activities={scenario['NUM_ACTIVITIES']} | Budget={scenario['BUDGET']}")

        # Generiraj aktivnosti
        activities = generate_data(scenario["NUM_ACTIVITIES"])

        # DEAP toolbox
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, scenario["NUM_ACTIVITIES"])
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)

        # Fitness funkcija po scenariju
        def single_objective_fitness_scenario(individual):
            total_cost = sum(activities[i]["cost"] for i, sel in enumerate(individual) if sel)
            total_roi = sum(activities[i]["roi"] for i, sel in enumerate(individual) if sel)
            if total_cost > scenario["BUDGET"]:
                return (-(total_cost - scenario["BUDGET"]),)
            return (total_roi,)

        def monte_carlo_eval_duration_scenario(individual):
            selected = [act for i, act in enumerate(activities) if individual[i] == 1]
            if not selected:
                return 0.0
            durations = [
                sum(random.triangular(act["optimistic"], act["pessimistic"], act["realistic"])
                    for act in selected)
                for _ in range(CONFIG["NUM_SIMULATIONS"])
            ]
            return np.mean(durations)

        toolbox.register("evaluate", single_objective_fitness_scenario)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Pokretanje GA
        for name, params in configs_ablation:
            print(f"--- GA konfiguracija: {name} ({CONFIG['RUNS']} ponavljanja) ---")
            for run in range(CONFIG["RUNS"]):
                pop = toolbox.population(n=params["pop_size"])
                hof = tools.HallOfFame(1)

                algorithms.eaSimple(
                    pop,
                    toolbox,
                    cxpb=params["cxpb"],
                    mutpb=params["mutpb"],
                    ngen=params["ngen"],
                    halloffame=hof,
                    verbose=False
                )

                best_ind = hof[0]
                final_roi = single_objective_fitness_scenario(best_ind)[0]
                final_duration = monte_carlo_eval_duration_scenario(best_ind)

                all_results.append({
                    "Scenario": scenario["name"],
                    "GA_Configuration": name,
                    "Run": run + 1,
                    "ROI": final_roi,
                    "Duration": final_duration
                })
                print(f"  Run {run+1}: ROI={final_roi:.3f}, Duration={final_duration:.3f}")

    # Spremanje svih rezultata
    df_all = pd.DataFrame(all_results)
    df_all.to_csv("ablation_study_all_scenarios.csv", index=False)
    print("\nSvi rezultati spremljeni u 'ablation_study_all_scenarios.csv'")

# ==============================================================================
# VIZUALIZACIJA
# ==============================================================================
def plot_ablation_boxplot_all_scenarios():
    """Vizualizira rezultate svih scenarija i GA konfiguracija iz CSV-a."""
    try:
        df = pd.read_csv("ablation_study_all_scenarios.csv")
    except FileNotFoundError:
        print("Datoteka 'ablation_study_all_scenarios.csv' nije pronađena.")
        return

    sns.set_theme(style="whitegrid")

    if not os.path.exists("graf_box"):
        os.makedirs("graf_box")

    scenarios = df["Scenario"].unique()
    for scenario in scenarios:
        df_scen = df[df["Scenario"] == scenario]
        order = ["Bez križanja", "Bez mutacije", "Standardni GA", "Više generacija", "Veća populacija"]

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f"Scenarij: {scenario} | Usporedba performansi GA", fontsize=18)

        sns.boxplot(ax=axes[0], x="GA_Configuration", y="ROI", data=df_scen, order=order, palette="viridis")
        axes[0].set_title("Distribucija ROI vrijednosti", fontsize=14)
        axes[0].set_xlabel("GA konfiguracija", fontsize=12)
        axes[0].set_ylabel("ROI", fontsize=12)
        axes[0].tick_params(axis="x", rotation=45)

        sns.boxplot(ax=axes[1], x="GA_Configuration", y="Duration", data=df_scen, order=order, palette="plasma")
        axes[1].set_title("Distribucija procijenjenog trajanja", fontsize=14)
        axes[1].set_xlabel("GA konfiguracija", fontsize=12)
        axes[1].set_ylabel("Prosječno trajanje (dani)", fontsize=12)
        axes[1].tick_params(axis="x", rotation=45)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_filename = f"graf_box/{scenario}_ga_boxplot.png"
        plt.savefig(output_filename, dpi=300)
        print(f"Kutijasti graf za scenarij '{scenario}' spremljen u '{output_filename}'")
        plt.close(fig)

# ==============================================================================
# GLAVNI POKRETAČ
# ==============================================================================
if __name__ == "__main__":
    start_time = time.time()

    run_ablation_study()
    plot_ablation_boxplot_all_scenarios()

    end_time = time.time()
    print(f"\nUkupno vrijeme izvođenja Eksperimenta: {end_time - start_time:.2f} sekundi.")

