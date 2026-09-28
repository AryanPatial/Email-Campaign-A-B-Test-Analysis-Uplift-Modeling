"""Step 3: can we email fewer people and keep most of the extra visits?

Idea: an uplift model guesses, for each customer, how much MORE likely they are
to visit if we email them. If a small group gets most of the lift, we only email them.

Method (called a "two-model" or T-learner approach):
  model_email    - trained on emailed customers, predicts P(visit | email)
  model_no_email - trained on control customers, predicts P(visit | no email)
  uplift         = model_email - model_no_email

I only use the Mens email here because step 2 showed it is the better email.

Run from the project folder:  python src/03_uplift_model.py
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

OUT = Path("outputs")
SEEDS = [1, 2, 3, 4, 5]          # 5 different train/test splits, so one lucky split can't fool me
TARGET_SHARES = [0.3, 0.5]       # "what if we only email the top 30% / 50%?"

con = duckdb.connect("data/email.duckdb", read_only=True)
df = con.execute("SELECT * FROM customers WHERE arm IN ('mens_email', 'control')").df()
con.close()

df["treated"] = (df["arm"] == "mens_email").astype(int)
X = pd.get_dummies(df[["recency", "history", "mens", "womens", "newbie", "zip_area", "channel"]],
                   drop_first=True).astype(float)
X["log_history"] = np.log(X["history"])
y = df["visit"]

MODELS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=3000),
    "gradient_boosting": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                                max_iter=200, random_state=0),
}


def extra_visits(part):
    """Extra visits caused by the email in this group of customers.

    Visit rate (emailed) minus visit rate (not emailed), times group size.
    This works because the email was assigned at random inside every group.
    """
    t, c = part[part["treated"] == 1], part[part["treated"] == 0]
    return (t["visit"].mean() - c["visit"].mean()) * len(part)


def share_captured(test, score_col, share):
    """If we email only the top `share` of customers by score, what share of the extra visits do we keep?"""
    top = test.nlargest(int(len(test) * share), score_col)
    return extra_visits(top) / extra_visits(test)


rows, decile_rows = [], []
for model_name, make_model in MODELS.items():
    for seed in SEEDS:
        train_idx, test_idx = train_test_split(df.index, test_size=0.3, random_state=seed,
                                               stratify=df["treated"])
        train, test = df.loc[train_idx], df.loc[test_idx].copy()
        is_t = train["treated"] == 1

        model_email = make_model().fit(X.loc[train.index[is_t]], y.loc[train.index[is_t]])
        model_no_email = make_model().fit(X.loc[train.index[~is_t]], y.loc[train.index[~is_t]])

        p_email = model_email.predict_proba(X.loc[test.index])[:, 1]
        p_no_email = model_no_email.predict_proba(X.loc[test.index])[:, 1]
        test["uplift_score"] = p_email - p_no_email
        # A normal "who is likely to visit?" model, for comparison.
        test["response_score"] = p_email

        test_t = test["treated"] == 1
        auc = roc_auc_score(test.loc[test_t, "visit"], test.loc[test_t, "response_score"])

        for share in TARGET_SHARES:
            rows.append({"model": model_name, "seed": seed, "email_top_share": share,
                         "uplift_model_keeps": share_captured(test, "uplift_score", share),
                         "response_model_keeps": share_captured(test, "response_score", share),
                         "random_keeps": share,
                         "response_model_auc": auc})

        # Actual lift in each tenth of customers, sorted from highest to lowest predicted uplift.
        test["decile"] = pd.qcut(test["uplift_score"].rank(method="first", ascending=False),
                                 10, labels=range(1, 11))
        for d, part in test.groupby("decile", observed=True):
            t, c = part[part["treated"] == 1], part[part["treated"] == 0]
            decile_rows.append({"model": model_name, "seed": seed, "decile": int(d),
                                "predicted_uplift": part["uplift_score"].mean(),
                                "actual_uplift": t["visit"].mean() - c["visit"].mean()})

results = pd.DataFrame(rows)
results.to_csv(OUT / "uplift_targeting_by_seed.csv", index=False)

summary = (results.groupby(["model", "email_top_share"])
           [["uplift_model_keeps", "response_model_keeps", "random_keeps", "response_model_auc"]]
           .agg(["mean", "min", "max"]).round(3))
pd.set_option("display.width", 200)
print("Share of extra visits kept if we only email the top X% (average over 5 splits, with min and max)")
print(summary.to_string())

deciles = (pd.DataFrame(decile_rows)
           .groupby(["model", "decile"])[["predicted_uplift", "actual_uplift"]].mean().round(4)
           .reset_index())
deciles.to_csv(OUT / "uplift_by_decile.csv", index=False)
print("\nPredicted vs actual uplift by decile (1 = highest predicted), averaged over splits")
print(deciles.to_string(index=False))
