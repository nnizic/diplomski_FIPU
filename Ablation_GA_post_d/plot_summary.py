"""
plot_summary.py
Generira summary grafove ROI i trajanja za sve scenarije iz CSV-a
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Putanja do CSV datoteke
CSV_FILE = "ablation_study_all_scenarios.csv"

# Folder za spremanje grafova
OUTPUT_FOLDER = "graf_box_summary"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Učitaj CSV
df = pd.read_csv(CSV_FILE)

# Postavi temu
sns.set_theme(style="whitegrid")

# ----- 1. Boxplot ROI po scenariju i GA konfiguraciji -----
plt.figure(figsize=(18, 8))
sns.boxplot(
    x="Scenario",
    y="ROI",
    hue="GA_Configuration",
    data=df,
    palette="viridis"
)
plt.title("Distribucija ROI po scenariju i GA konfiguraciji", fontsize=16)
plt.xlabel("Scenarij", fontsize=12)
plt.ylabel("ROI", fontsize=12)
plt.xticks(rotation=45)
plt.legend(title="GA konfiguracija")
plt.tight_layout()
roi_filename = f"{OUTPUT_FOLDER}/summary_ROI.png"
plt.savefig(roi_filename, dpi=300)
plt.close()
print(f"Summary ROI graf spremljen u '{roi_filename}'")

# ----- 2. Boxplot Duration po scenariju i GA konfiguraciji -----
plt.figure(figsize=(18, 8))
sns.boxplot(
    x="Scenario",
    y="Duration",
    hue="GA_Configuration",
    data=df,
    palette="plasma"
)
plt.title("Distribucija procijenjenog trajanja po scenariju i GA konfiguraciji", fontsize=16)
plt.xlabel("Scenarij", fontsize=12)
plt.ylabel("Prosječno trajanje (dani)", fontsize=12)
plt.xticks(rotation=45)
plt.legend(title="GA konfiguracija")
plt.tight_layout()
duration_filename = f"{OUTPUT_FOLDER}/summary_Duration.png"
plt.savefig(duration_filename, dpi=300)
plt.close()
print(f"Summary Duration graf spremljen u '{duration_filename}'")

print("\nSvi summary grafovi su generirani.")

