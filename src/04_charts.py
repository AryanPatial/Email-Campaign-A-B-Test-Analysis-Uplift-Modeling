"""Step 4: charts for the README and the slide deck. Reads the CSVs made by steps 1-3.

Run from the project folder:  python src/04_charts.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("outputs")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

GRAY, BLUE, ORANGE = "#8a8984", "#2a78d6", "#eb6834"
INK, MUTED = "#0b0b0b", "#52514e"
ARM_COLOR = {"control": GRAY, "mens_email": BLUE, "womens_email": ORANGE}
ARM_LABEL = {"control": "No email", "mens_email": "Mens email", "womens_email": "Womens email"}

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e5e0", "axes.axisbelow": True,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK,
})

# --- Chart 1: visit rate by group, with 95% CI ---
arms = pd.read_csv(OUT / "arm_summary.csv").set_index("arm").loc[list(ARM_LABEL)]
rate = arms["visit_rate"]
ci = 1.96 * np.sqrt(rate * (1 - rate) / arms["customers"])
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.bar([ARM_LABEL[a] for a in arms.index], rate * 100, yerr=ci * 100, capsize=6,
       color=[ARM_COLOR[a] for a in arms.index], width=0.55, error_kw={"ecolor": INK, "lw": 1})
for i, r in enumerate(rate):
    ax.text(i, r * 100 + 1.0, f"{r:.1%}", ha="center", color=INK, fontweight="bold")
ax.set_ylabel("Visited the site within 2 weeks (%)")
ax.set_ylim(0, 22)
ax.grid(axis="x", visible=False)
ax.set_title("The Mens email lifted visits from 10.6% to 18.3%")
fig.text(0.01, 0.01, "About 21,300 customers per group. Bars show 95% confidence intervals.",
         color=MUTED, fontsize=9)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "1_visit_rate_by_group.png", dpi=150)
plt.close(fig)

# --- Chart 2: lift by buyer type ---
seg = pd.read_csv(OUT / "lift_by_buyer_type.csv")
seg = seg[seg["metric"] == "visit"]
buyer_order = ["mens_only", "womens_only", "both"]
buyer_label = ["Bought men's only", "Bought women's only", "Bought both"]
x = np.arange(len(buyer_order))
fig, ax = plt.subplots(figsize=(7.5, 4.4))
for offset, arm in [(-0.18, "mens_email"), (0.18, "womens_email")]:
    part = seg[seg["arm"] == arm].set_index("buyer_type").loc[buyer_order]
    err = [(part["diff"] - part["ci_low"]) * 100, (part["ci_high"] - part["diff"]) * 100]
    ax.bar(x + offset, part["diff"] * 100, width=0.34, yerr=err, capsize=4,
           color=ARM_COLOR[arm], label=ARM_LABEL[arm], edgecolor="white", linewidth=2,
           error_kw={"ecolor": INK, "lw": 1})
ax.set_xticks(x, buyer_label)
ax.set_ylabel("Extra visits vs no email (percentage points)")
ax.grid(axis="x", visible=False)
ax.legend(frameon=False, loc="upper left")
ax.annotate("Womens email barely\nmoves men-only buyers\n(+1.1 pts)", xy=(0.18, 1.3),
            xytext=(0.55, 9.5), color=INK, fontsize=9,
            arrowprops={"arrowstyle": "->", "color": MUTED})
ax.set_title("The Mens email works for every buyer type")
fig.tight_layout()
fig.savefig(FIG / "2_lift_by_buyer_type.png", dpi=150)
plt.close(fig)

# --- Chart 3: predicted vs actual uplift by decile (logistic regression) ---
dec = pd.read_csv(OUT / "uplift_by_decile.csv")
dec = dec[dec["model"] == "logistic_regression"]
fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.bar(dec["decile"], dec["actual_uplift"] * 100, color=BLUE, width=0.7,
       edgecolor="white", linewidth=2, label="Actual lift (held-out test data)")
ax.plot(dec["decile"], dec["predicted_uplift"] * 100, color=INK, marker="o", markersize=6,
        lw=2, label="Model's predicted lift")
ax.axhline(7.66, color=MUTED, ls="--", lw=1)
ax.text(10.45, 7.66, "average\n7.7 pts", color=MUTED, fontsize=9, va="center")
ax.set_xticks(range(1, 11))
ax.set_xlabel("Customers split into 10 groups, highest predicted lift (1) to lowest (10)")
ax.set_ylabel("Extra visits from the email (pts)")
ax.set_xlim(0.4, 11.3)
ax.grid(axis="x", visible=False)
ax.legend(frameon=False, loc="upper right")
ax.set_title("Even the 'worst' customers still respond - targeting saves little")
fig.text(0.01, 0.01, "Average of 5 random 70/30 train/test splits. Mens email vs no email.",
         color=MUTED, fontsize=9)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "3_uplift_by_decile.png", dpi=150)
plt.close(fig)

print("Saved charts to", FIG)
