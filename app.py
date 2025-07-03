# ----------------------------------------------------------------------
# FocusNest Dashboard  –  syllabus-aligned (Table 2.4) with extras
# ----------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
import streamlit as st

from utils import load_data, get_numeric_df   # tidy_column_names is used inside utils

# ----------------------------------------------------------------------
# Basic page config
# ----------------------------------------------------------------------
st.set_page_config(page_title="FocusNest Dashboard",
                   page_icon="🪺", layout="wide")
st.sidebar.image("assets/FocusNest_logo.png", width=180)
st.sidebar.title("FocusNest")
st.sidebar.write("Build Better Habits. Break the Social Cycle.")

# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
DATA_PATH = "data/Dataset_for_Business_AssociationRuleReady.xlsx"
df = load_data(DATA_PATH)          # column names already tidied/lower-cased

# ----------------------------------------------------------------------
# Sidebar global filters
# ----------------------------------------------------------------------
age_min, age_max = st.sidebar.slider("Age range", 18, 60, (18, 60))
gender_filter = st.sidebar.multiselect(
    "Gender",
    options=df["gender"].unique().tolist(),
    default=df["gender"].unique().tolist()
)

df_view = df.query(
    "age >= @age_min and age <= @age_max and gender in @gender_filter"
)

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tabs = st.tabs(["📊 Visualisation", "🤖 Classification",
                "👥 Clustering", "🔗 Assoc Rules", "📈 Regression"])

# =====================================================================
# 1. VISUALISATION TAB
# =====================================================================
with tabs[0]:
    st.header("Data Exploration & Insights")
    sns.set_style("whitegrid")

    # KPI cards --------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)

    avg_minutes = df_view["daily_minutes_spent"].mean()
    heavy_pct   = (df_view["daily_minutes_spent"] > 180).mean() * 100
    avg_income  = df_view["monthly_income"].mean()
    avg_posts   = df_view["posts_per_day"].mean() if "posts_per_day" in df_view.columns else np.nan

    k1.metric("Avg Daily Minutes", f"{avg_minutes:.1f}")
    k2.metric("% Heavy Users (>180 min)", f"{heavy_pct:.1f}%")
    k3.metric("Avg Monthly Income", f"${avg_income:,.0f}")
    if not np.isnan(avg_posts):
        k4.metric("Avg Posts/Day", f"{avg_posts:.1f}")

    # Age histogram & Minutes KDE -------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Age Distribution")
        fig = px.histogram(df_view, x="age", nbins=15,
                           color_discrete_sequence=["#bca43a"])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Minutes Spent (KDE)")
        fig, ax = plt.subplots()
        sns.kdeplot(df_view["daily_minutes_spent"], ax=ax, shade=True,
                    color="#bca43a")
        st.pyplot(fig)

    st.dataframe(df_view.head())

    # Box-plot Pay vs Willingness -------------------------------------
    st.markdown("#### Pay vs Subscription Intent")
    fig = px.box(df_view, x="willingness_to_subscribe", y="pay_amount",
                 color="willingness_to_subscribe",
                 color_discrete_sequence=["#f9d278", "#d6b55b", "#bca43a"])
    st.plotly_chart(fig, use_container_width=True)

    # Platform bar-chart ----------------------------------------------
    st.markdown("#### Platform Usage Share")
    platform_cols = [c for c in df.columns if c.startswith("uses_")]
    plat_counts = (df_view[platform_cols]
                   .sum()
                   .rename(lambda c: c.replace("uses_", ""))
                   .sort_values(ascending=False))
    fig = px.bar(plat_counts, x=plat_counts.index, y=plat_counts.values,
                 color=plat_counts.values, color_continuous_scale="YlOrBr",
                 labels={"x": "Platform", "y": "Users"})
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heat-map --------------------------------------------
    st.markdown("#### Correlation Heat-map")
    corr = df_view.select_dtypes("number").corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, cmap="YlOrBr", ax=ax)
    st.pyplot(fig)

# =====================================================================
# 2. CLASSIFICATION TAB
# =====================================================================
with tabs[1]:
    st.header("Classification")

    # Hyper-parameter sliders
    k_val = st.slider("K for KNN", 3, 15, 5, step=2)
    tree_depth = st.slider("Max depth for Decision Tree", 2, 10, 3)

    # ---------- Data prep ----------
    X = get_numeric_df(df)

    # tidy & map target to 0/1/2, then drop any unmapped rows
    y = (
        df["willingness_to_subscribe"]
        .astype(str).str.strip().str.lower()          # clean text
        .map({"no": 0, "maybe": 1, "yes": 2})
    )
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )


    models = {
        "KNN": KNeighborsClassifier(n_neighbors=k_val),
        "Decision Tree": DecisionTreeClassifier(max_depth=tree_depth, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "GBRT": GradientBoostingClassifier(random_state=42),
    }

    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        results[name] = [accuracy_score(y_test, pred),
                         precision_score(y_test, pred, average="macro"),
                         recall_score(y_test, pred, average="macro"),
                         f1_score(y_test, pred, average="macro")]

    st.dataframe(pd.DataFrame(results, index=["Acc", "Prec", "Rec", "F1"]).T)

      # ---------- Confusion matrix (table + heat-map) ----------
    cm_choice = st.selectbox("Confusion-Matrix model", list(models.keys()))
    if st.button("Show Confusion Matrix"):
        cm = confusion_matrix(y_test, models[cm_choice].predict(X_test))

        # 1️⃣ tidy table --------------------------------------------------
        classes = ["No", "Maybe", "Yes"]
        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual {c}" for c in classes],
            columns=[f"Pred {c}" for c in classes]
        )
        st.subheader("Confusion-Matrix Table")
        st.dataframe(cm_df.style.background_gradient(cmap="YlOrBr"))

        # 2️⃣ heat-map ----------------------------------------------------
        st.subheader("Confusion-Matrix Heat-map")
        fig, ax = plt.subplots()
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="YlOrBr", ax=ax,
                    cbar=False, linewidths=.5, linecolor="white")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)


# =====================================================================
# 3. CLUSTERING TAB
# =====================================================================
with tabs[2]:
    st.header("Clustering (K-means)")
    k = st.slider("k (clusters)", 2, 10, 4)
    Xnum = get_numeric_df(df)
    km = KMeans(n_clusters=k, random_state=42).fit(Xnum)
    df["cluster"] = km.labels_

    if st.checkbox("Show Elbow Curve"):
        inertias = [KMeans(n_clusters=i, random_state=42).fit(Xnum).inertia_
                    for i in range(2, 11)]
        fig, ax = plt.subplots()
        ax.plot(range(2, 11), inertias, marker="o")
        ax.set_xlabel("k"); ax.set_ylabel("Inertia")
        st.pyplot(fig)

    fig = px.scatter(df, x="age", y="daily_minutes_spent",
                     color="cluster", symbol="gender",
                     color_continuous_scale="YlOrBr")
    st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# 4. ASSOCIATION RULES TAB
# =====================================================================
with tabs[3]:
    st.header("Association Rule Mining")

    binary_cols = [c for c in df.columns if set(df[c].unique()) <= {0, 1}]
    cols_sel = st.multiselect("Binary columns (min 2)", binary_cols,
                              default=binary_cols[:2])
    support_val = st.slider("min_support", 0.01, 0.2, 0.05, 0.01)
    conf_val    = st.slider("min_confidence", 0.10, 0.90, 0.30, 0.05)

    if len(cols_sel) >= 2:
        frequent = apriori(df[cols_sel], min_support=support_val,
                           use_colnames=True)
        rules = association_rules(frequent, metric="confidence",
                                  min_threshold=conf_val)
        st.dataframe(rules[["antecedents", "consequents",
                            "support", "confidence", "lift"]].head(10))

        if st.checkbox("Show Rule Network (top 15)"):
            top = rules.sort_values("lift", ascending=False).head(15)
            G = nx.DiGraph()
            for _, r in top.iterrows():
                for a in r["antecedents"]:
                    for c in r["consequents"]:
                        G.add_edge(a, c, weight=r["lift"])
            fig, ax = plt.subplots(figsize=(6, 4))
            pos = nx.spring_layout(G, seed=42, k=0.5)
            nx.draw_networkx(G, pos, ax=ax, node_color="#ffda66",
                             node_size=500, font_size=6,
                             edge_color="#9e8b3a")
            st.pyplot(fig)

# =====================================================================
# 5. REGRESSION TAB  (Linear · Ridge · Lasso · Tuned Decision-Tree)
# =====================================================================
with tabs[4]:
    st.header("Regression Models")

    # Choose target
    target_choices = ["pay_amount", "daily_minutes_spent"]
    if "sleep_quality" in df.columns:
        target_choices.append("sleep_quality")
    target = st.selectbox("Select target variable", target_choices)

    # Feature / target split
    numeric_df = get_numeric_df(df)
    Xreg = numeric_df.drop(columns=[target])
    yreg = numeric_df[target]

    # Tune Decision-Tree depth
    depths = list(range(2, 9))
    cv_scores = [cross_val_score(
                    DecisionTreeRegressor(max_depth=d, random_state=42),
                    Xreg, yreg, cv=5, scoring="r2").mean()
                 for d in depths]
    depth_df = pd.DataFrame({"max_depth": depths, "CV R²": cv_scores})
    best_depth = int(depth_df.loc[depth_df["CV R²"].idxmax(), "max_depth"])
    st.dataframe(depth_df.style.background_gradient(cmap="YlOrBr"), height=200)

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        f"DecisionTree (depth={best_depth})":
            DecisionTreeRegressor(max_depth=best_depth, random_state=42)
    }

    results, preds = {}, {}
    for name, m in models.items():
        m.fit(Xreg, yreg)
        pred = m.predict(Xreg)
        preds[name] = pred
        results[name] = {
            "R²": m.score(Xreg, yreg),
            "RMSE": np.sqrt(((pred - yreg) ** 2).mean())
        }

    st.dataframe(pd.DataFrame(results).T.style.background_gradient(cmap="YlOrBr"))

    best_name = max(results, key=lambda k: results[k]["R²"])
    best_pred = preds[best_name]
    st.markdown(f"##### Best model: **{best_name}**")

    fig = px.scatter(x=yreg, y=best_pred,
                     labels={"x": "Actual", "y": "Predicted"},
                     title=f"Actual vs Predicted — {best_name}",
                     color_discrete_sequence=["#bca43a"])
    fig.add_shape(type="line", x0=yreg.min(), x1=yreg.max(),
                  y0=yreg.min(), y1=yreg.max(),
                  line=dict(dash="dash", color="#7e7309"))
    st.plotly_chart(fig, use_container_width=True)

    fig, ax = plt.subplots()
    sns.residplot(x=best_pred, y=yreg - best_pred,
                  color="#bca43a", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Residuals")
    st.pyplot(fig)
    st.caption("Residual plot checks bias/heteroscedasticity.")
