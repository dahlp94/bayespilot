import numpy as np
import pandas as pd

np.random.seed(42)

n = 500

usage = np.random.normal(150, 50, n).clip(10, 400)
bill = usage * 0.6 + np.random.normal(0, 20, n)
support_calls = np.random.poisson(2, n)

regions = np.random.choice(["north", "south", "east", "west"], size=n)

# TRUE underlying relationship (hidden ground truth)
logit = (
    0.02 * usage
    + 0.04 * bill
    + 0.5 * support_calls
    - 10
)

prob = 1 / (1 + np.exp(-logit))

churn = np.random.binomial(1, prob)

df = pd.DataFrame({
    "usage": usage.round(2),
    "bill": bill.round(2),
    "support_calls": support_calls,
    "region": regions,
    "churn": churn
})

df.to_csv("datasets/churn.csv", index=False)

print("Dataset saved to datasets/churn.csv")