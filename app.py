"""
Streamlit + Seaborn demo on the *tips* dataset
(1) Interactive EDA plots            – sidebar selectbox
(2) Simple linear regression demo    – button
      target  : tip
      feature : total_bill
"""

import streamlit as st

# ── MUST be first Streamlit command ───────────────────────────
st.set_page_config(page_title="Tips demo", layout="centered")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.linear_model import LinearRegression
import numpy as np

# ──────────────────────────────────────────────────────────────
# 1. Data loading helper (works online or offline)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_tips() -> pd.DataFrame:
    try:
        return sns.load_dataset("tips")          # pulls from GitHub
    except Exception:                            # fallback for offline runs
        _CSV = """
total_bill,tip,sex,smoker,day,time,size
16.99,1.01,Female,No,Sun,Dinner,2
10.34,1.66,Male,No,Sun,Dinner,3
21.01,3.50,Male,No,Sun,Dinner,3
23.68,3.31,Male,No,Sun,Dinner,2
24.59,3.61,Female,No,Sun,Dinner,4
25.29,4.71,Male,No,Sun,Dinner,4
26.88,3.12,Male,No,Sun,Dinner,4
15.04,1.96,Male,No,Sun,Dinner,2
14.78,3.23,Male,No,Sun,Dinner,2
20.65,3.35,Male,No,Sat,Dinner,3
"""
        return pd.read_csv(StringIO(_CSV.strip()))

tips = load_tips()

# ──────────────────────────────────────────────────────────────
# 2. Streamlit layout
# ──────────────────────────────────────────────────────────────
st.title("Seaborn × Streamlit – *tips* dataset")

plot_kind = st.sidebar.selectbox(
    "Choose a plot",
    ("Scatter: total_bill vs tip",
     "Box: tip by day",
     "Histogram: total_bill")
)

# ──────────────────────────────────────────────────────────────
# 3. Draw the selected plot
# ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
sns.set_theme(style="whitegrid")

if plot_kind.startswith("Scatter"):
    sns.scatterplot(
        data=tips, x="total_bill", y="tip",
        hue="sex", style="smoker", s=100, ax=ax
    )
    ax.set_title("Tip vs. Total Bill")

elif plot_kind.startswith("Box"):
    sns.boxplot(
        data=tips, x="day", y="tip",
        hue="smoker", ax=ax
    )
    ax.set_title("Tip by Day (Smoker vs Non-smoker)")

else:  # Histogram
    sns.histplot(
        tips["total_bill"], kde=True, bins=15, ax=ax
    )
    ax.set_title("Distribution of Total Bills")

st.pyplot(fig)

st.markdown("---")
st.dataframe(tips.head())

# ──────────────────────────────────────────────────────────────
# 4. Linear regression demo (tip ~ total_bill)
# ──────────────────────────────────────────────────────────────
if st.button("Run linear regression (tip ~ total_bill)"):
    X = tips[["total_bill"]].values
    y = tips["tip"].values
    model = LinearRegression().fit(X, y)

    coef = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(X, y)

    st.subheader("Regression results")
    st.write(f"**Equation:**  tip = {intercept:.2f} + {coef:.2f} × total_bill")
    st.write(f"**R²:** {r2:.3f}")

    # Plot scatter with regression line
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sns.scatterplot(x=X.flatten(), y=y, ax=ax2)
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_range)
    ax2.plot(x_range, y_pred, color="red", linewidth=2)
    ax2.set_xlabel("total_bill")
    ax2.set_ylabel("tip")
    ax2.set_title("Linear regression: tip vs total_bill")
    st.pyplot(fig2)
