"""Eksperimenti - Diplomski rad - Neven Nižić"""

from functools import partial
import random

import numpy as np
import pandas as pd

from deap import algorithms, base, creator, tools

# ==============================================================================
# PRIVREMENI KONFIGURACIJSKI RJEČNIK (SAMO ZA TESTIRANJE)
# ==============================================================================
# CONFIG = {
#    "SEED": 42,
#    "RUNS": 1,  # <<< SMANJENO NA 1 ZA BRZI TEST
#    "NUM_SIMULATIONS": 100,
#    "experimental_series": [
#        # Svi ostali eksperimenti su zakomentirani ili obrisani
#        {
#            "name": "A3_Slozeni",
#            "NUM_ACTIVITIES": 100,
#            "BUDGET": 5000,
#            "POP_SIZE": 250,
#            "NGEN": 200,
#            "CX_PB": 0.7,
#            "MUT_PB": 0.2,
#        },
#    ],
# }

# ==============================================================================
# GLAVNI KONFIGURACIJSKI RJEČNIK
# ==============================================================================
# Ovdje su definirani SVE eksperimente koje želite provesti.
# Skripta će automatski iterirati kroz ovu listu.


CONFIG = {
    "SEED": 42,
    "RUNS": 10,  # Broj ponavljanja za statističku značajnost
    "NUM_SIMULATIONS": 100,  # Broj iteracija za Monte Carlo procjenu trajanja
    "experimental_series": [
        # Serija A: Testiranje Skalabilnosti
        {
            "name": "A1_Osnovni",
            "NUM_ACTIVITIES": 10,
            "BUDGET": 1000,
            "POP_SIZE": 100,
            "NGEN": 40,
            "CX_PB": 0.7,
            "MUT_PB": 0.2,
        },
        {
            "name": "A2_Srednji",
            "NUM_ACTIVITIES": 50,
            "BUDGET": 2500,
            "POP_SIZE": 200,
            "NGEN": 150,
            "CX_PB": 0.7,
            "MUT_PB": 0.2,
        },
        {
            "name": "A3_Slozeni",
            "NUM_ACTIVITIES": 100,
            "BUDGET": 5000,
            "POP_SIZE": 250,
            "NGEN": 200,
            "CX_PB": 0.7,
            "MUT_PB": 0.2,
        },
        # Serija B: Testiranje Utjecaja Ograničenja (koristi istu složenost kao A2)
        {
            "name": "B1_Restriktivan",
            "NUM_ACTIVITIES": 50,
            "BUDGET": 1500,
            "POP_SIZE": 200,
            "NGEN": 150,
            "CX_PB": 0.7,
            "MUT_PB": 0.2,
        },
        {
            "name": "B3_Labav",
            "NUM_ACTIVITIES": 50,
            "BUDGET": 4000,
            "POP_SIZE": 200,
            "NGEN": 150,
            "CX_PB": 0.7,
            "MUT_PB": 0.2,
        },
    ],
}

# Inicijalizacija generatora slučajnih brojeva
random.seed(CONFIG["SEED"])
np.random.seed(CONFIG["SEED"])

# Kreiranje DEAP tipova (jednom na početku)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("IndividualMulti", list, fitness=creator.FitnessMulti)

# ==============================================================================
# POMOĆNE FUNKCIJE (sada primaju 'config' i 'activities')
# ==============================================================================


def generate_data(config):
    """Generira slučajne aktivnosti na temelju konfiguracije."""
    return [
        {
            "id": i,
            "cost": random.randint(50, 200),
            "optimistic": random.randint(5, 10),
            "realistic": random.randint(10, 20),
            "pessimistic": random.randint(20, 40),
            "roi": round(random.uniform(1.0, 3.0), 2),
        }
        for i in range(config["NUM_ACTIVITIES"])
    ]


def calculate_metrics(individual, activities):
    """Vraća ukupni trošak i ROI."""
    total_cost = sum(activities[i]["cost"] for i, sel in enumerate(individual) if sel)
    total_roi = sum(activities[i]["roi"] for i, sel in enumerate(individual) if sel)
    return total_cost, total_roi


def monte_carlo_eval_duration(individual, activities, config):
    """Računa prosječno trajanje pomoću Monte Carlo simulacije."""
    selected = [act for i, act in enumerate(activities) if individual[i] == 1]
    if not selected:
        return 0.0
    durations = [
        sum(
            random.triangular(act["optimistic"], act["realistic"], act["pessimistic"])
            for act in selected
        )
        for _ in range(config["NUM_SIMULATIONS"])
    ]
    return np.mean(durations)


# ==============================================================================
# FITNESS FUNKCIJE (sada primaju 'config' i 'activities')
# ==============================================================================


def single_objective_fitness(individual, activities, config):
    """Jedno-objektivni cilj: maksimizirati ROI."""
    total_cost, total_roi = calculate_metrics(individual, activities)
    if total_cost > config["BUDGET"]:
        return (-(total_cost - config["BUDGET"]),)
    return (total_roi,)


def multi_objective_fitness(individual, activities, config):
    """Više-objektivni cilj: maksimizirati ROI, minimizirati trajanje."""
    total_cost, total_roi = calculate_metrics(individual, activities)
    if total_cost > config["BUDGET"]:
        return 0, 99999
    avg_duration = monte_carlo_eval_duration(individual, activities, config)
    return total_roi, avg_duration


# ==============================================================================
# FUNKCIJE ZA POKRETANJE SCENARIJA (sada primaju 'config' i 'activities')
# ==============================================================================


def run_random_search_once(config, activities):
    """Random Search - traži najbolje rješenje slučajnim generiranjem."""
    best_ind, best_roi = None, -1
    num_evaluations = config["POP_SIZE"] * config["NGEN"]
    for _ in range(num_evaluations):
        ind = [random.randint(0, 1) for _ in range(config["NUM_ACTIVITIES"])]
        cost, roi = calculate_metrics(ind, activities)
        if cost <= config["BUDGET"] and roi > best_roi:
            best_ind, best_roi = ind, roi
    if best_ind is None:
        return 0, 0
    return best_roi, monte_carlo_eval_duration(best_ind, activities, config)


def run_ga_once(
    config,
    activities,
    individual_type,
    fitness_func,
    selection_func,
    algorithm_func,
    return_logbook=False,  # NOVI ARGUMENT
    **kwargs,
):
    """Generička funkcija za pokretanje jedne instance GA."""
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register(
        "individual",
        tools.initRepeat,
        individual_type,
        toolbox.attr_bool,
        config["NUM_ACTIVITIES"],
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    evaluate_partial = partial(fitness_func, activities=activities, config=config)
    toolbox.register("evaluate", evaluate_partial)
    toolbox.register("select", selection_func, **kwargs)

    pop = toolbox.population(n=config["POP_SIZE"])
    hof = (
        tools.HallOfFame(1)
        if algorithm_func == algorithms.eaSimple
        else tools.ParetoFront()
    )

    # NOVI DIO: Inicijalizacija statistike ako je zatraženo
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Inicijalizacija logbook-a
    logbook = tools.Logbook()

    # Pokretanje odgovarajućeg DEAP algoritma
    # DODAN 'stats' ARGUMENT U POZIV ALGORITMA
    if algorithm_func == algorithms.eaSimple:
        pop, logbook = algorithms.eaSimple(
            pop,
            toolbox,
            cxpb=config["CX_PB"],
            mutpb=config["MUT_PB"],
            ngen=config["NGEN"],
            stats=stats,
            halloffame=hof,
            verbose=False,
        )
    else:  # eaMuPlusLambda
        pop, logbook = algorithms.eaMuPlusLambda(
            pop,
            toolbox,
            mu=config["POP_SIZE"],
            lambda_=config["POP_SIZE"],
            cxpb=config["CX_PB"],
            mutpb=config["MUT_PB"],
            ngen=config["NGEN"],
            stats=stats,
            halloffame=hof,
            verbose=False,
        )

    if not hof:
        # Ako nema rješenja, vraćamo prazne vrijednosti
        if return_logbook:
            return (0, 0), None
        return 0, 0

    # NOVI DIO: Vraćanje logbook-a ako je zatraženo
    if return_logbook:
        best_ind = hof[0]
        fitness_values = single_objective_fitness(best_ind, activities, config)
        duration = monte_carlo_eval_duration(best_ind, activities, config)
        return (fitness_values[0], duration), logbook

    # Postojeća logika za standardni povrat
    if algorithm_func == algorithms.eaSimple:
        best_ind = hof[0]
        fitness_values = single_objective_fitness(best_ind, activities, config)
        return fitness_values[0], monte_carlo_eval_duration(
            best_ind, activities, config
        )
    else:  # NSGA-II
        best_solution = max(hof, key=lambda ind: ind.fitness.values[0])
        return best_solution.fitness.values, hof


# ==============================================================================
# GLAVNI PROGRAM ZA PROVOĐENJE SVIH EKSPERIMENATA
# ==============================================================================


def run_full_study():
    """Glavna funkcija koja orkestrira sve eksperimente definirane u CONFIG-u."""
    master_results = []

    # Iteriramo kroz konfiguracije definirane u listi 'experimental_series'
    for exp_config in CONFIG["experimental_series"]:

        # Kreiranje kompletne konfiguracije za ovaj specifični eksperiment
        # spajanjem globalnih postavki (RUNS, NUM_SIMULATIONS) i lokalnih.
        config = {
            "RUNS": CONFIG["RUNS"],
            "NUM_SIMULATIONS": CONFIG["NUM_SIMULATIONS"],
            **exp_config,
        }

        print(f"\n===== ZAPOČINJEM EKSPERIMENT: {config['name']} =====")
        print(f"Korištena konfiguracija: {config}")

        # Za svaki eksperiment generiraju se novi, odgovarajući podatci
        activities = generate_data(config)

        # Scenariji koriste 'config' koji sadrži SVE potrebne parametre
        scenarios = {
            "Random Search (MC)": lambda: run_random_search_once(config, activities),
            "GA (samo ROI)": lambda: run_ga_once(
                config,
                activities,
                creator.Individual,
                single_objective_fitness,
                tools.selTournament,
                algorithms.eaSimple,
                tournsize=3,
            ),
            "GA+MC (NSGA-II)": lambda: run_ga_once(
                config,
                activities,
                creator.IndividualMulti,
                multi_objective_fitness,
                tools.selNSGA2,
                algorithms.eaMuPlusLambda,
            ),
        }

        for name, run_function in scenarios.items():
            print(f"--- Pokrećem scenarij: {name} ({config['RUNS']} puta) ---")
            run_rois, run_durations = [], []

            for i in range(config["RUNS"]):
                # NOVI DIO: Posebna logika za hvatanje logbook-a
                # Okida se samo za prvo pokretanje (i==0) A3_Slozeni GA (samo ROI) scenarija
                if (
                    config["name"] == "A3_Slozeni"
                    and name == "GA (samo ROI)"
                    and i == 0
                ):
                    print("    -> Bilježim statistike konvergencije...")
                    # Pozivamo funkciju direktno s 'return_logbook=True'
                    (roi, duration), logbook = run_ga_once(
                        config,
                        activities,
                        creator.Individual,
                        single_objective_fitness,
                        tools.selTournament,
                        algorithms.eaSimple,
                        return_logbook=True,  # Postavljamo zastavicu
                        tournsize=3,
                    )

                    # Spremanje logbook-a u CSV
                    df_log = pd.DataFrame(logbook)
                    df_log.to_csv("konvergencija_A3.csv", index=False)
                    print(
                        "    -> Podaci o konvergenciji spremljeni u 'konvergencija_A3.csv'"
                    )

                # Standardno pokretanje za sve ostale slučajeve
                else:
                    result = run_function()
                    if name == "GA+MC (NSGA-II)":
                        (roi, duration), _ = result  # Zanemarujemo pareto_front ovdje
                    else:
                        roi, duration = result

                run_rois.append(roi)
                run_durations.append(duration)
                print(
                    f"  Run {i+1}/{config['RUNS']}: ROI={roi:.2f}, Trajanje={duration:.2f}"
                )

            # Ostatak koda za agregiranje rezultata
            master_results.append(
                {
                    "Eksperiment": config["name"],
                    "Scenarij": name,
                    "ROI_mean": np.mean(run_rois),
                    "ROI_std": np.std(run_rois),
                    "Trajanje_mean": np.mean(run_durations),
                    "Trajanje_std": np.std(run_durations),
                }
            )
            print("-" * 50)

    # Kreiraj i spremi konačni DataFrame sa svim rezultatima
    df = pd.DataFrame(master_results)
    df.to_csv("master_rezultati.csv", index=False)

    print("\n\n===== SVI EKSPERIMENTI SU ZAVRŠENI =====")
    print("Konačni rezultati svih eksperimenata:")
    print(df.to_string())
    print("\nRezultati spremljeni u 'master_rezultati.csv'")


if __name__ == "__main__":
    run_full_study()
