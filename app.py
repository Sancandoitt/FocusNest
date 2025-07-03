# ------------- FocusNest Streamlit Dashboard (Table-2.4 visuals only) -------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from mlxtend.frequent_patterns import apriori, association_rules
from utils import load_data, get_numeric_df

# ---------------- basic page set-up ----------------
st.set_page_config(page_title="FocusNest Dashboard", page_icon="🪺", layout="wide")
st.sidebar.image("assets/FocusNest_logo.png", width=180)
st.sidebar.title("FocusNest")
st.sidebar.write("Build Better Habits. Break the Social Cycle.")

DATA_PATH = "data/Dataset_for_Business_AssociationRuleReady.xlsx"
df = load_data(DATA_PATH)

# ---------------- global sidebar filters ----------------
age_min, age_max = st.sidebar.slider("Age range", 18, 60, (18, 60))
gender_filter = st.sidebar.multiselect(
    "Gender",
    options=df["Gender"].unique().tolist(),
    default=df["Gender"].unique().tolist(),
)
df_view = df.query("Age >= @age_min and Age <= @age_max and Gender in @gender_filter")

# ---------------- tabs ----------------
tabs = st.tabs(
    ["📊 Visualisation", "🤖 Classification", "👥 Clustering",
     "🔗 Assoc Rules", "📈 Regression"]
)

# =======================================================================
# 1. VISUALISATION TAB  (only methods taught – Table 2.4)
# =======================================================================
with tabs[0]:
    st.header("Data Exploration & Insights (visuals from Table 2.4)")
    sns.set_style("whitegrid")

    # KPI cards ----------------------------------------------------------
    k1, k2, k3 = st.columns(3)
    k1.metric("Avg Daily Minutes",  f"{df_view['Daily_Minutes_Spent'].mean():.1f}")
    k2.metric("% Heavy Users (>180 min)",
              f"{(df_view['Daily_Minutes_Spent']>180).mean()*100:.1f}%")
    k3.metric("Avg Monthly Income", f"${df_view['Monthly_Income'].mean():,.0f}")

    # Two basic plots (hist & KDE) ---------------------------------------
    c1, c2 = st.columns(2)

    with c1:  # Age histogram
        st.subheader("Age Distribution")
        fig = px.histogram(df_view, x="Age", nbins=15,
                           color_discrete_sequence=["#bca43a"])
        st.plotly_chart(fig, use_container_width=True)

    with c2:  # Minutes KDE
        st.subheader("Minutes Spent (KDE)")
        fig, ax = plt.subplots()
        sns.kdeplot(df_view["Daily_Minutes_Spent"], shade=True, ax=ax)
        st.pyplot(fig)

    st.dataframe(df_view.head())  # quick peek

    # Box-plot Pay × Willingness ----------------------------------------
    st.markdown("#### Pay Amount vs Subscription Intent (box-plot)")
    fig = px.box(
        df_view,
        x="Willingness_to_Subscribe",
        y="Pay_Amount",
        color="Willingness_to_Subscribe",
        color_discrete_sequence=["#f9d278", "#d6b55b", "#bca43a"],
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Clear price-sensitivity bands: ‘Yes’ cohort median ≈ $10+; "
        "‘Maybe’ ≈ $2-5; ‘No’ zero — informs tier pricing."
    )

    # Bar-chart Platform counts -----------------------------------------
    st.markdown("#### Platform Usage Share (bar-chart)")
    platform_cols = [c for c in df.columns if c.startswith("Uses_")]
    plat_counts = (
        df_view[platform_cols]
        .sum()
        .rename(lambda c: c.replace("Uses_", ""))
        .sort_values(ascending=False)
    )
    fig = px.bar(
        plat_counts,
        x=plat_counts.index,
        y=plat_counts.values,
        color=plat_counts.values,
        labels={"x": "Platform", "y": "User Count", "color": "Users"},
        color_continuous_scale="YlOrBr",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Instagram & TikTok dominate — focus acquisition there first.")

    # Correlation heat-map ----------------------------------------------
    st.markdown("#### Numeric Correlation Heat-map")
    num_cols = df_view.select_dtypes("number").columns
    corr = df_view[num_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, cmap="YlOrBr", ax=ax)
    st.pyplot(fig)
    st.caption("Dark cells reveal strong relationships; guides feature selection.")

# =======================================================================
# 2. CLASSIFICATION TAB
# =======================================================================
with tabs[1]:
    st.header("Classification")
    X = get_numeric_df(df)
    y = df["Willingness_to_Subscribe"].map({"No": 0, "Maybe": 1, "Yes": 2})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "GBRT": GradientBoostingClassifier(),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        results[name] = [
            accuracy_score(y_test, pred),
            precision_score(y_test, pred, average="macro"),
            recall_score(y_test, pred, average="macro"),
            f1_score(y_test, pred, average="macro"),
        ]

    st.dataframe(pd.DataFrame(results, index=["Acc", "Prec", "Rec", "F1"]).T)

    # Confusion matrix + optional Decision-tree plot
    choice = st.selectbox("Confusion Matrix model", list(models.keys()))
    if st.button("Show Confusion Matrix"):
        cm = confusion_matrix(y_test, models[choice].predict(X_test))
        st.write(cm)

     if st.checkbox("Show Decision-Tree (first 2 levels)"):
         fig, ax = plt.subplots(figsize=(6, 4))
         plot_tree(models["Decision Tree"], max_depth=2,
                   feature_names=X.columns, filled=True, ax=ax, fontsize=6)
         st.pyplot(fig)

# =======================================================================
# 3. CLUSTERING TAB  (scatter & elbow – taught methods)
# =======================================================================
with tabs[2]:
    st.header("Clustering (K-means)")
    k = st.slider("k (clusters)", 2, 10, 4)
    Xnum = get_numeric_df(df)
    km = KMeans(n_clusters=k, random_state=42).fit(Xnum)
    df["Cluster"] = km.labels_

    # Elbow curve
    if st.checkbox("Show Elbow Curve"):
        inertias = [KMeans(n_clusters=i, random_state=42).fit(Xnum).inertia_
                    for i in range(2, 11)]
        fig, ax = plt.subplots()
        ax.plot(range(2, 11), inertias, marker="o")
        ax.set_xlabel("k")
        ax.set_ylabel("Inertia")
        st.pyplot(fig)

    # Scatter Age vs Minutes coloured by cluster
    fig = px.scatter(
        df, x="Age", y="Daily_Minutes_Spent",
        color="Cluster", symbol="Gender",
        color_continuous_scale="YlOrBr"
    )
    st.plotly_chart(fig, use_container_width=True)

# =======================================================================
# 4. ASSOCIATION RULES TAB  (table + optional network)
# =======================================================================
with tabs[3]:
    st.header("Association Rule Mining")
    binary_cols = [c for c in df.columns if set(df[c].unique()) <= {0, 1}]
    columns_sel = st.multiselect("Binary columns (min 2)", binary_cols, default=binary_cols[:2])
    support_val = st.slider("min_support", 0.01, 0.2, 0.05)

    if len(columns_sel) >= 2:
        frequent = apriori(df[columns_sel], min_support=support_val, use_colnames=True)
        rules = association_rules(frequent, metric="confidence", min_threshold=0.3)
        st.dataframe(rules[["antecedents", "consequents", "support",
                            "confidence", "lift"]].head(10))

        # optional simple network
        import networkx as nx
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
                             node_size=500, font_size=6, edge_color="#9e8b3a")
            st.pyplot(fig)

# =======================================================================
# 5. REGRESSION TAB  (simple comparison)
# =======================================================================
with tabs[4]:
    st.header("Regression (predict Pay_Amount)")
    target = "Pay_Amount"
    Xreg = get_numeric_df(df).drop(columns=[target])
    yreg = df[target]
    regs = {"Linear": LinearRegression(), "Ridge": Ridge(), "Lasso": Lasso()}
    res = {}
    for name, model in regs.items():
        model.fit(Xreg, yreg)
        res[name] = model.score(Xreg, yreg)
    st.dataframe(pd.Series(res, name="R²").to_frame())
