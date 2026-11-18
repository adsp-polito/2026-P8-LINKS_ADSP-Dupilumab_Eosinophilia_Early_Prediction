"""Plot baseline numeric features to compare mean, median, and missingness."""

from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "Datasets" / "Processed").is_dir())
SOURCE = next((ROOT / "Datasets" / "Processed").glob("*Anagrafica.csv"))
OUTPUT = ROOT / "reports" / "numeric_distributions"
OUTPUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "Età", "Peso (kg)", "Altezza (cm)", "BMI", "Num Comorbidità",
    "Anni di astinenza", "Siga al giorno", "Numero di brevi cicli/anno di OCS",
    "Dosaggio cumulativo annuale in mg di prednisone",
    "Numero di pregressi interventi ESS", "N. allergeni positivi",
    "Giorni al mese di utilizzo spray", "ACT", "NPS", "Neutrofili",
    "Linfociti", "Monociti", "Eosinofili", "Basofili", "IgE totali",
    "Physical Symptoms", "Health and QoL", "Nasal Symptoms",
    "Otologic Symptoms", "Sleep Symptoms", "Emotional Symptoms", "NCS",
    "Ostruzione nasale", "Rinorrea anteriore", "Rinorrea posteriore",
    "Dolore facciale", "Starnuti", "QoL",
]

df = pd.read_csv(SOURCE, dtype=str, encoding="utf-8-sig")
df = df.loc[pd.to_numeric(df["ID paziente"], errors="coerce").notna()].copy()
missing_columns = set(FEATURES) - set(df.columns)
if missing_columns:
    raise KeyError(f"Colonne mancanti: {sorted(missing_columns)}")

numeric = df[FEATURES].apply(
    lambda col: pd.to_numeric(col.str.strip().str.replace(",", ".", regex=False), errors="coerce")
)
smoking = pd.to_numeric(df["Fumo Baseline (0=no, 1=ex, 2=si)"], errors="coerce")
numeric.loc[smoking.eq(2), "Anni di astinenza"] = 0
numeric.loc[smoking.eq(0), "Anni di astinenza"] = numeric.loc[smoking.eq(0), "Età"]

stats = pd.DataFrame({
    "n": numeric.notna().sum(),
    "missing": numeric.isna().sum(),
    "media": numeric.mean(),
    "mediana": numeric.median(),
    "min": numeric.min(),
    "max": numeric.max(),
})
stats.index.name = "feature"
stats.to_csv(OUTPUT / "riepilogo.csv", encoding="utf-8-sig")

for page, start in enumerate(range(0, len(FEATURES), 9), start=1):
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for ax, feature in zip(axes.flat, FEATURES[start:start + 9]):
        values = numeric[feature].dropna()
        if values.empty:
            ax.text(0.5, 0.5, "Nessun valore disponibile", ha="center", va="center")
        else:
            ax.hist(values, bins=min(20, max(5, math.ceil(math.sqrt(len(values))))),
                    color="#7ba7c2", edgecolor="white")
            ax.axvline(values.mean(), color="#c0392b", linewidth=2,
                       label=f"Media: {values.mean():.2f}")
            ax.axvline(values.median(), color="#1b6e3c", linewidth=2,
                       linestyle="--", label=f"Mediana: {values.median():.2f}")
            ax.legend(fontsize=8)
        ax.set_title(feature, fontsize=10)
        ax.set_xlabel(f"n={len(values)}  |  mancanti={numeric[feature].isna().sum()}", fontsize=9)
        ax.set_ylabel("Pazienti")
    for ax in list(axes.flat)[len(FEATURES[start:start + 9]):]:
        ax.axis("off")
    fig.suptitle(f"Distribuzioni delle feature numeriche — pagina {page}", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"distribuzioni_{page}.png", dpi=160)
    plt.close(fig)

print(f"Creati {math.ceil(len(FEATURES) / 9)} grafici in {OUTPUT}")
