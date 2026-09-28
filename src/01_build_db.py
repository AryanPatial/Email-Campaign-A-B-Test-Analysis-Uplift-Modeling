"""Step 1: load the data into DuckDB, run the data checks, and check the test was set up fairly.

Run from the project folder:  python src/01_build_db.py
"""
from pathlib import Path

import duckdb
import pandas as pd
from scipy import stats

DB_PATH = "data/email.duckdb"
OUT = Path("outputs")


def run_sql_file(con, path):
    con.execute(Path(path).read_text())


con = duckdb.connect(DB_PATH)
run_sql_file(con, "sql/01_load_clean.sql")

# --- 1. Data checks (the SQL returns one row per check) ---
checks = con.execute(Path("sql/02_checks.sql").read_text()).df()
print("Data checks")
print(checks.to_string(index=False))
if checks["bad_rows"].sum() > 0:
    raise SystemExit("A data check failed. Stopping.")

# Rows that look exactly the same. There is no customer id in the raw file,
# so two different customers can have the same values. I keep them.
dupes = con.execute("""
    SELECT count(*) - count(DISTINCT (arm, recency, history, mens, womens, zip_area,
                                      newbie, channel, visit, conversion, spend))
    FROM customers
""").fetchone()[0]
print(f"\nRows identical to another row: {dupes:,} (kept - no id to tell them apart)")

# --- 2. Sample ratio check ---
# The company meant to split customers into 3 equal groups.
# If the counts are far from equal, the split was broken and the results can't be trusted.
counts = con.execute("SELECT arm, count(*) AS n FROM customers GROUP BY arm ORDER BY arm").df()
chi2, p_srm = stats.chisquare(counts["n"])
print("\nGroup sizes")
print(counts.to_string(index=False))
print(f"Sample ratio check: chi-square p = {p_srm:.3f} (above 0.01 means the split looks fine)")

# --- 3. Balance check ---
# Random assignment should make the groups look the same BEFORE the email.
# Standardized mean difference (SMD) = gap in means / pooled std. Under 0.1 is the usual bar.
df = con.execute("SELECT * FROM customers").df()
df["web"] = (df["channel"] == "Web").astype(int)
df["urban"] = (df["zip_area"] == "Urban").astype(int)
control = df[df["arm"] == "control"]

rows = []
for arm in ["mens_email", "womens_email"]:
    treated = df[df["arm"] == arm]
    for col in ["recency", "history", "mens", "womens", "newbie", "web", "urban"]:
        pooled_sd = ((treated[col].var() + control[col].var()) / 2) ** 0.5
        smd = (treated[col].mean() - control[col].mean()) / pooled_sd
        rows.append({"arm": arm, "feature": col,
                     "treated_mean": round(treated[col].mean(), 3),
                     "control_mean": round(control[col].mean(), 3),
                     "smd": round(smd, 3)})

balance = pd.DataFrame(rows)
print("\nBalance check (largest |SMD| per arm)")
print(balance.assign(abs_smd=balance["smd"].abs())
      .groupby("arm")["abs_smd"].max().to_string())

# --- 4. Summary tables for the A/B test and the dashboard ---
run_sql_file(con, "sql/03_summary.sql")
OUT.mkdir(exist_ok=True)
con.execute("SELECT * FROM arm_summary").df().to_csv(OUT / "arm_summary.csv", index=False)
con.execute("SELECT * FROM buyer_type_summary").df().to_csv(OUT / "buyer_type_summary.csv", index=False)
balance.to_csv(OUT / "balance_check.csv", index=False)

print("\nResults by arm")
print(con.execute("SELECT * FROM arm_summary").df().to_string(index=False))
con.close()
