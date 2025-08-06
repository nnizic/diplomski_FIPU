"""Vizualizacija podataka - diplomski rad - Neven Nižić"""

import os

import matplotlib.pyplot as plt
import pandas as pd

# 📌 Globalne postavke (lako mijenjati na jednom mjestu)
CSV_FILE = "usporedni_rezultati.csv"
OUTPUT_DIR = "slike"

COL_SCENARIO = "Scenarij"
COL_ROI_MEAN = "ROI_mean"
COL_ROI_STD = "ROI_std"
COL_DURATION_MEAN = "Trajanje_mean"
COL_DURATION_STD = "Trajanje_std"

# Boje za stupce (koliko postavki - toliko boja)
BAR_COLORS = ["skyblue", "orange", "green", "red", "purple"]


def plot_results():
    # Učitaj CSV rezultate
    df = pd.read_csv(CSV_FILE)

    # Kreiraj folder za slike ako ne postoji
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1️⃣ Graf ROI-a
    plt.figure(figsize=(8, 5))
    plt.bar(df[COL_SCENARIO], df[COL_ROI_MEAN], color=BAR_COLORS[: len(df)])
    plt.ylabel("ROI (mean)")
    plt.title("Usporedba ROI-a po postavci")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/usporedba_roi.png", dpi=300)
    plt.close()

    # 2️⃣ Graf prosječnog trajanja
    plt.figure(figsize=(8, 5))
    plt.bar(df[COL_SCENARIO], df[COL_DURATION_MEAN], color=BAR_COLORS[: len(df)])
    plt.ylabel("Trajanje (dani, mean)")
    plt.title("Usporedba prosječnog trajanja projekta po postavci")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/usporedba_trajanje.png", dpi=300)
    plt.close()

    print(f"📊 Grafovi spremljeni u mapu '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    plot_results()
