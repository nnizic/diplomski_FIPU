"""Vizualizacija _ diplomski rad _ Neven Nižić"""

#####################################################
#
# Modul za vizualizaciju - dio za vizualizaciju konvergencije
#
####################################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_convergence():
    """Učitava podatke o konvergenciji i sprema grafikon."""
    try:
        df_log = pd.read_csv("konvergencija_A3.csv")
    except FileNotFoundError:
        print(
            "Datoteka 'konvergencija_A3.csv' nije pronađena. Jeste li pokrenuli glavni eksperiment?"
        )
        return

    # Postavljanje stila
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Crtanje linija
    sns.lineplot(
        x="gen",
        y="max",
        data=df_log,
        label="Maksimalni ROI (Najbolja jedinka)",
        color="blue",
        linewidth=2.5,
    )
    sns.lineplot(
        x="gen",
        y="avg",
        data=df_log,
        label="Prosječni ROI (Cijela populacija)",
        color="orange",
        linestyle="--",
        linewidth=2,
    )

    # Uređivanje grafikona
    plt.title("Grafikon konvergencije za eksperiment A3 (GA samo ROI)", fontsize=16)
    plt.xlabel("Generacija", fontsize=12)
    plt.ylabel("Povrat na investiciju (ROI)", fontsize=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    plt.tight_layout()

    # Spremanje slike
    output_filename = "grafikoni_final/E_konvergencija_A3.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Grafikon konvergencije spremljen u '{output_filename}'")


if __name__ == "__main__":
    plot_convergence()
