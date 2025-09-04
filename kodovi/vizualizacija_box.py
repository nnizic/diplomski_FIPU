"""Vizualizacija kutijastog grafa kalibracije GA - diplomski rad - Neven Nižić"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ==============================================================================
# VIZUALIZACIJA (Kutijasti grafovi kao odvojene slike)
# ==============================================================================
def plot_ablation_boxplots_separate():
    """
    Učitava sirove rezultate ablacijske studije i stvara DVIJE ODVOJENE SLIKE
    s kutijastim grafovima: jednu za ROI, jednu za Trajanje.
    """
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

    # --- PRVA SLIKA: ROI ---
    plt.figure(figsize=(10, 7))  # Stvaramo novu, odvojenu figuru
    ax_roi = sns.boxplot(
        x="Konfiguracija", y="ROI", data=df, order=order, palette="viridis"
    )

    ax_roi.set_title("Distribucija ROI vrijednosti po konfiguracijama", fontsize=16)
    ax_roi.set_xlabel("Konfiguracija GA", fontsize=12)
    ax_roi.set_ylabel("ROI", fontsize=12)
    plt.xticks(rotation=45)  # Rotiramo labele radi bolje čitljivosti
    plt.tight_layout()

    # Spremanje prve slike
    output_filename_roi = "graf_box/ga_boxplot_roi.png"
    plt.savefig(output_filename_roi, dpi=300)
    print(f"Graf za ROI spremljen u '{output_filename_roi}'")
    plt.close()  # Zatvaramo figuru da ne smeta sljedećoj

    # --- DRUGA SLIKA: Trajanje ---
    plt.figure(figsize=(10, 7))  # Stvaramo drugu, odvojenu figuru
    ax_dur = sns.boxplot(
        x="Konfiguracija", y="Trajanje", data=df, order=order, palette="plasma"
    )

    ax_dur.set_title(
        "Distribucija procijenjenog trajanja po konfiguracijama", fontsize=16
    )
    ax_dur.set_xlabel("Konfiguracija GA", fontsize=12)
    ax_dur.set_ylabel("Prosječno trajanje (dani)", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Spremanje druge slike
    output_filename_dur = "graf_box/ga_boxplot_trajanje.png"
    plt.savefig(output_filename_dur, dpi=300)
    print(f"Graf za trajanje spremljen u '{output_filename_dur}'")
    plt.close()


# ==============================================================================
# GLAVNI POKRETAČ
# ==============================================================================
if __name__ == "__main__":
    plot_ablation_boxplots_separate()
