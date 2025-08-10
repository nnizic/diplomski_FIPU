import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 📌 GLOBALNE POSTAVKE
# ==============================================================================
# Naziv datoteke s rezultatima
CSV_FILE = "master_rezultati.csv"

# Folder u koji će se spremati slike
OUTPUT_DIR = "grafikoni_final"

# Definicija boja i markera za svaki scenarij radi konzistentnosti
PALETTE = {
    "Random Search (MC)": "gray",
    "GA (samo ROI)": "royalblue",
    "GA+MC (NSGA-II)": "darkorange",
}
MARKERS = {
    "Random Search (MC)": "s",  # square
    "GA (samo ROI)": "o",  # circle
    "GA+MC (NSGA-II)": "D",  # diamond
}

# ==============================================================================
# FUNKCIJE ZA CRTANJE GRAFIKONA
# ==============================================================================


def plot_scalability_roi(df):
    """Generira linijski grafikon za analizu skalabilnosti (ROI vs. Složenost)."""
    print("Kreiram grafikon: Skalabilnost - ROI...")

    # 1. Filtriranje podataka samo za Seriju A
    series_a_experiments = ["A1_Osnovni", "A2_Srednji", "A3_Slozeni"]
    df_series_a = df[df["Eksperiment"].isin(series_a_experiments)].copy()

    # 2. Osiguravamo ispravan redoslijed na x-osi
    df_series_a["Eksperiment"] = pd.Categorical(
        df_series_a["Eksperiment"], categories=series_a_experiments, ordered=True
    )

    # 3. Crtanje
    plt.figure(figsize=(12, 7))
    sns.lineplot(
        data=df_series_a,
        x="Eksperiment",
        y="ROI_mean",
        hue="Scenarij",
        style="Scenarij",
        palette=PALETTE,
        markers=MARKERS,
        markersize=10,
        linewidth=2.5,
    )

    plt.title(
        "Usporedba ROI-a s povećanjem složenosti problema (Serija A)",
        fontsize=16,
        pad=20,
    )
    plt.xlabel("Složenost eksperimenta", fontsize=12)
    plt.ylabel("Prosječni ROI (mean)", fontsize=12)
    plt.legend(title="Optimizacijski scenarij", frameon=True)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "A_skalabilnost_roi.png"), dpi=300)
    plt.close()


def plot_scalability_duration(df):
    """Generira linijski grafikon za analizu skalabilnosti (Trajanje vs. Složenost)."""
    print("Kreiram grafikon: Skalabilnost - Trajanje...")

    # 1. Filtriranje i sortiranje (isto kao za ROI)
    series_a_experiments = ["A1_Osnovni", "A2_Srednji", "A3_Slozeni"]
    df_series_a = df[df["Eksperiment"].isin(series_a_experiments)].copy()
    df_series_a["Eksperiment"] = pd.Categorical(
        df_series_a["Eksperiment"], categories=series_a_experiments, ordered=True
    )

    # 2. Crtanje
    plt.figure(figsize=(12, 7))
    sns.lineplot(
        data=df_series_a,
        x="Eksperiment",
        y="Trajanje_mean",
        hue="Scenarij",
        style="Scenarij",
        palette=PALETTE,
        markers=MARKERS,
        markersize=10,
        linewidth=2.5,
    )

    plt.title(
        "Usporedba trajanja projekta s povećanjem složenosti (Serija A)",
        fontsize=16,
        pad=20,
    )
    plt.xlabel("Složenost eksperimenta", fontsize=12)
    plt.ylabel("Prosječno trajanje (dani, mean)", fontsize=12)
    plt.legend(title="Optimizacijski scenarij")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "A_skalabilnost_trajanje.png"), dpi=300)
    plt.close()


def plot_budget_impact_roi(df):
    """Generira stupčasti dijagram za analizu utjecaja budžeta na ROI."""
    print("Kreiram grafikon: Utjecaj budžeta - ROI...")

    # 1. Filtriranje podataka za Seriju B
    series_b_experiments = ["B1_Restriktivan", "A2_Srednji", "B3_Labav"]
    df_series_b = df[df["Eksperiment"].isin(series_b_experiments)].copy()

    # Uklanjamo neuspjele rezultate za NSGA-II radi čišće vizualizacije
    # Ovi neuspjesi se obrađuju i diskutiraju u tekstualnom dijelu rada
    df_series_b = df_series_b[df_series_b["ROI_mean"] > 0]

    # 2. Osiguravamo ispravan redoslijed na x-osi
    df_series_b["Eksperiment"] = pd.Categorical(
        df_series_b["Eksperiment"], categories=series_b_experiments, ordered=True
    )

    # 3. Crtanje
    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=df_series_b, x="Eksperiment", y="ROI_mean", hue="Scenarij", palette=PALETTE
    )

    plt.title("Utjecaj restriktivnosti budžeta na ROI (Serija B)", fontsize=16, pad=20)
    plt.xlabel("Postavka budžeta", fontsize=12)
    plt.ylabel("Prosječni ROI (mean)", fontsize=12)
    plt.legend(title="Optimizacijski scenarij")
    plt.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "B_budzet_roi.png"), dpi=300)
    plt.close()


def plot_stability_analysis(df):
    """Generira stupčasti dijagram za analizu stabilnosti (standardne devijacije)."""
    print("Kreiram grafikon: Analiza stabilnosti...")

    # Koristimo sve eksperimente osim neuspjelog B1 za NSGA-II
    df_stability = df[
        ~(
            (df["Eksperiment"] == "B1_Restriktivan")
            & (df["Scenarij"] == "GA+MC (NSGA-II)")
        )
    ]

    # 1. Crtanje stabilnosti ROI-a
    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=df_stability, x="Eksperiment", y="ROI_std", hue="Scenarij", palette=PALETTE
    )
    plt.title(
        "Analiza stabilnosti rješenja: Standardna devijacija ROI-a", fontsize=16, pad=20
    )
    plt.xlabel("Eksperiment", fontsize=12)
    plt.ylabel("Standardna devijacija ROI-a (std)", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "C_stabilnost_roi.png"), dpi=300)
    plt.close()

    # 2. Crtanje stabilnosti Trajanja
    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=df_stability,
        x="Eksperiment",
        y="Trajanje_std",
        hue="Scenarij",
        palette=PALETTE,
    )
    plt.title(
        "Analiza stabilnosti rješenja: Standardna devijacija Trajanja",
        fontsize=16,
        pad=20,
    )
    plt.xlabel("Eksperiment", fontsize=12)
    plt.ylabel("Standardna devijacija Trajanja (std)", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "C_stabilnost_trajanje.png"), dpi=300)
    plt.close()


# ==============================================================================
# GLAVNA FUNKCIJA
# ==============================================================================


def main():
    """Glavna funkcija koja učitava podatke i poziva funkcije za crtanje."""
    print("--- Započinjem proces vizualizacije rezultata ---")

    # Provjeri postoji li CSV datoteka
    if not os.path.exists(CSV_FILE):
        print(
            f"GREŠKA: Datoteka '{CSV_FILE}' nije pronađena. Molim prvo pokrenite skriptu za eksperimente."
        )
        return

    # Učitaj podatke
    df = pd.read_csv(CSV_FILE)
    print(f"Učitano {len(df)} redaka iz '{CSV_FILE}'.")

    # Kreiraj folder za slike ako ne postoji
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Slike će biti spremljene u folder: '{OUTPUT_DIR}'")

    # Pozovi funkcije za generiranje svih grafikona
    plot_scalability_roi(df)
    plot_scalability_duration(df)
    plot_budget_impact_roi(df)
    plot_stability_analysis(df)

    print("\n--- Proces vizualizacije je uspješno završen! ---")


if __name__ == "__main__":
    main()
