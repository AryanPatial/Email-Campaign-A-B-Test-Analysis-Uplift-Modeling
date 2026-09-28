"""Step 2: the A/B test. Did the emails work, by how much, and for whom?

Run from the project folder:  python src/02_ab_test.py
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

OUT = Path("outputs")
rng = np.random.default_rng(42)
N_BOOT = 2000

con = duckdb.connect("data/email.duckdb", read_only=True)
df = con.execute("SELECT * FROM customers").df()
con.close()


def bootstrap_diff_ci(a, b, n_boot=N_BOOT):
    """95% CI for mean(a) - mean(b), by resampling each group with replacement.

    I use this for spend because spend is mostly zeros with a few big orders,
    so I don't want to lean only on the normal-curve formula.
    """
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean()
    return np.percentile(diffs, [2.5, 97.5])


def compare(a, b, metric):
    """Compare group a vs group b on one metric. Returns diff, 95% CI and p-value."""
    diff = a.mean() - b.mean()
    if metric in ("visit", "conversion"):
        # Two-proportion z-test. CI uses the unpooled standard error.
        se = np.sqrt(a.mean() * (1 - a.mean()) / len(a) + b.mean() * (1 - b.mean()) / len(b))
        p_pool = (a.sum() + b.sum()) / (len(a) + len(b))
        se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / len(a) + 1 / len(b)))
        p_value = 2 * stats.norm.sf(abs(diff / se_pool))
        low, high = diff - 1.96 * se, diff + 1.96 * se
    else:
        # Welch t-test for the p-value, bootstrap for the CI.
        p_value = stats.ttest_ind(a, b, equal_var=False).pvalue
        low, high = bootstrap_diff_ci(a, b)
    return diff, low, high, p_value


# --- 1. Main results: each email vs no email, and the two emails vs each other ---
pairs = [("mens_email", "control"), ("womens_email", "control"), ("mens_email", "womens_email")]
metrics = ["visit", "conversion", "spend"]
n_tests = len(pairs) * len(metrics)
alpha = 0.05 / n_tests  # Bonferroni: 9 tests, so each one needs p < 0.0056

rows = []
for arm_a, arm_b in pairs:
    for metric in metrics:
        a = df.loc[df["arm"] == arm_a, metric].to_numpy()
        b = df.loc[df["arm"] == arm_b, metric].to_numpy()
        diff, low, high, p = compare(a, b, metric)
        rows.append({"comparison": f"{arm_a} vs {arm_b}", "metric": metric,
                     "rate_a": a.mean(), "rate_b": b.mean(),
                     "diff": diff, "ci_low": low, "ci_high": high,
                     "relative_lift": diff / b.mean(), "p_value": p,
                     "significant_after_bonferroni": p < alpha})

results = pd.DataFrame(rows)
results.to_csv(OUT / "ab_test_results.csv", index=False)
pd.set_option("display.width", 160)
print(f"Main results (Bonferroni cut-off p < {alpha:.4f})")
print(results.round(4).to_string(index=False))

# --- 2. Money: extra revenue from the emails in the 2-week window ---
print("\nExtra revenue in the 2 weeks after the email (vs no email)")
revenue_rows = []
for arm in ["mens_email", "womens_email"]:
    a = df.loc[df["arm"] == arm, "spend"].to_numpy()
    b = df.loc[df["arm"] == "control", "spend"].to_numpy()
    diff = a.mean() - b.mean()
    low, high = bootstrap_diff_ci(a, b)
    revenue_rows.append({"arm": arm, "customers_emailed": len(a),
                         "extra_revenue": diff * len(a),
                         "ci_low": low * len(a), "ci_high": high * len(a)})
revenue = pd.DataFrame(revenue_rows)
revenue.to_csv(OUT / "extra_revenue.csv", index=False)
print(revenue.round(0).to_string(index=False))

# --- 3. Regression: same answer, with pre-email features added ---
# Adding features that predict visits (but were fixed before the email) should not
# move the estimate much - the test is randomized - but it can shrink the CI.
# HC1 = standard errors that don't assume equal variance.
df["mens_email"] = (df["arm"] == "mens_email").astype(int)
df["womens_email"] = (df["arm"] == "womens_email").astype(int)
plain = smf.ols("visit ~ mens_email + womens_email", data=df).fit(cov_type="HC1")
adjusted = smf.ols("visit ~ mens_email + womens_email + recency + np.log(history)"
                   " + mens + womens + newbie + C(channel) + C(zip_area)",
                   data=df).fit(cov_type="HC1")
print("\nRegression on visit: effect of the Mens email")
for name, model in [("no features", plain), ("with features", adjusted)]:
    est, se = model.params["mens_email"], model.bse["mens_email"]
    print(f"  {name:14s} estimate {est:.4f}   std error {se:.5f}")

# --- 4. For whom? Lift split by what the customer bought last year ---
seg_rows = []
for buyer in ["mens_only", "womens_only", "both"]:
    part = df[df["buyer_type"] == buyer]
    for arm in ["mens_email", "womens_email"]:
        for metric in ["visit", "spend"]:
            a = part.loc[part["arm"] == arm, metric].to_numpy()
            b = part.loc[part["arm"] == "control", metric].to_numpy()
            diff, low, high, p = compare(a, b, metric)
            seg_rows.append({"buyer_type": buyer, "arm": arm, "metric": metric,
                             "customers_in_arm": len(a), "diff": diff,
                             "ci_low": low, "ci_high": high, "p_value": p})
segments = pd.DataFrame(seg_rows)
segments.to_csv(OUT / "lift_by_buyer_type.csv", index=False)
print("\nLift by buyer type (difference vs no email in the same buyer type)")
print(segments.round(4).to_string(index=False))

# Is the Womens email's lift really smaller for men-only buyers, or is that noise?
# Interaction test: the womens_email:mens_only term is the gap in lift.
sub = df[df["arm"].isin(["womens_email", "control"])].copy()
sub["mens_only"] = (sub["buyer_type"] == "mens_only").astype(int)
inter = smf.ols("visit ~ womens_email * mens_only", data=sub).fit(cov_type="HC1")
term = "womens_email:mens_only"
print(f"\nInteraction test (Womens email x men-only buyer): "
      f"{inter.params[term]:.4f}, 95% CI {inter.conf_int().loc[term, 0]:.4f} "
      f"to {inter.conf_int().loc[term, 1]:.4f}, p = {inter.pvalues[term]:.2g}")
