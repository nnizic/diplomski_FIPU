import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_expanded_ablation_results():
    """
    Učitava rezultate proširene ablacijske studije i stvara
    odvojene slike s kutijastim grafovima za svaki scenarij.
    """
    try:
        df = pd.read_csv("ablation_study_all_scenarios.csv") # Pretpostavljam da ste datoteku tako nazvali
    except FileNotFoundError:
        print("Datoteka 'ablation_study_all_scenarios.csv' nije pronađena.")
        return

    # Kreiranje foldera za spremanje slika ako ne postoji
    output_dir = "slike/ablation_by_scenario"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scenarios = df['Scenario'].unique()
    order = ["Bez križanja", "Bez mutacije", "Standardni GA", "Više generacija", "Veća populacija"]
    sns.set_theme(style="whitegrid")

    # Prolazimo kroz svaki scenarij i stvaramo zasebnu sliku
    for scenario in scenarios:
        print(f"Generiram graf za scenarij: {scenario}...")
        
        # Filtriramo DataFrame samo za trenutni scenarij
        df_scenario = df[df['Scenario'] == scenario]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f'Analiza GA konfiguracija za scenarij: {scenario}', fontsize=18)

        # --- PRVI GRAF: ROI ---
        sns.boxplot(ax=axes[0], x="GA_Configuration", y="ROI", data=df_scenario, order=order, palette="viridis")
        axes[0].set_title('Distribucija ROI vrijednosti', fontsize=14)
        axes[0].set_xlabel('Konfiguracija GA', fontsize=12)
        axes[0].set_ylabel('ROI', fontsize=12)
        axes[0].tick_params(axis='x', rotation=45)

        # --- DRUGI GRAF: Trajanje ---
        sns.boxplot(ax=axes[1], x="GA_Configuration", y="Duration", data=df_scenario, order=order, palette="plasma")
        axes[1].set_title('Distribucija procijenjenog trajanja', fontsize=14)
        axes[1].set_xlabel('Konfiguracija GA', fontsize=12)
        axes[1].set_ylabel('Prosječno trajanje (dani)', fontsize=12)
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Spremanje slike
        output_filename = os.path.join(output_dir, f"ablation_plot_{scenario}.png")
        plt.savefig(output_filename, dpi=300)
        plt.close() # Zatvaramo figuru
        print(f" -> Spremljeno u '{output_filename}'")

# --- KAKO KORISTITI ---
# 2. Pokrenite ovu funkciju
if __name__ == "__main__":
    plot_expanded_ablation_results()
