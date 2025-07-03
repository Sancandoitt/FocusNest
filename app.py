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
# 1. VISUALISATION TAB  
# =======================================================================
with tabs[0]:
    st.header("Data Exploration & Insights")
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

    # ---------- Hyper-parameter sliders ----------
    k_val = st.slider("K for KNN", 3, 15, 5, step=2)
    tree_depth = st.slider("Max depth for Decision Tree", 2, 10, 3)

    # ---------- Data prep ----------
    X = get_numeric_df(df)
    y = df["Willingness_to_Subscribe"].map({"No": 0, "Maybe": 1, "Yes": 2})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # ---------- Model dict (uses slider values) ----------
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=k_val),
        "Decision Tree": DecisionTreeClassifier(max_depth=tree_depth, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "GBRT": GradientBoostingClassifier(random_state=42),
    }

    # ---------- Train & evaluate ----------
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

    st.dataframe(pd.DataFrame(results,
                              index=["Acc", "Prec", "Rec", "F1"]).T)

    # ---------- Confusion matrix ----------
    cm_choice = st.selectbox("Confusion Matrix model", list(models.keys()))
    if st.button("Show Confusion Matrix"):
        cm = confusion_matrix(y_test, models[cm_choice].predict(X_test))
        st.write(cm)

    # ---------- Decision-Tree diagram (optional) ----------
    if st.checkbox("Show Decision-Tree (first 2 levels)"):
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_tree(
            models["Decision Tree"],
            feature_names=X.columns,
            max_depth=2,
            filled=True,
            ax=ax,
            fontsize=6,
        )
        st.pyplot(fig)


# =======================================================================
# 3. CLUSTERING TAB  
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
# 4. ASSOCIATION RULES TAB
# =======================================================================
with tabs[3]:
    st.header("Association Rule Mining")

    binary_cols = [c for c in df.columns if set(df[c].unique()) <= {0, 1}]
    columns_sel = st.multiselect(
        "Binary columns (min 2)",
        binary_cols,
        default=binary_cols[:2]
    )

    support_val = st.slider("min_support", 0.01, 0.2, 0.05, 0.01)
    conf_val    = st.slider("min_confidence", 0.10, 0.90, 0.30, 0.05)

    if len(columns_sel) >= 2:
        frequent = apriori(
            df[columns_sel],
            min_support=support_val,
            use_colnames=True
        )
        rules = association_rules(
            frequent,
            metric="confidence",
            min_threshold=conf_val
        )

        st.dataframe(
            rules[["antecedents", "consequents", "support",
                   "confidence", "lift"]].head(10)
        )

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
            nx.draw_networkx(
                G, pos, ax=ax,
                node_color="#ffda66",
                node_size=500,
                font_size=6,
                edge_color="#9e8b3a"
            )
            st.pyplot(fig)
# =======================================================================


# =======================================================================
# 5. REGRESSION TAB  +  Decision-Tree hyper-parameter tuning
# =======================================================================
with tabs[4]:
    st.header("Regression Models")

    # ---------- choose target ----------
  target_choices = ["pay_amount",
                  "daily_minutes_spent",
                  "sleep_quality"]          # ← new
    target = st.selectbox("Select target variable", target_choices)

    # ---------- feature matrix ----------
    numeric_df = get_numeric_df(df)
    Xreg = numeric_df.drop(columns=[target])
    yreg = numeric_df[target]

    # ---------- base models ----------
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import cross_val_score

    base_models = {
        "Linear" : LinearRegression(),
        "Ridge"  : Ridge(),
        "Lasso"  : Lasso(),
    }

    # ---------- tune Decision-Tree on max_depth ----------
    depths = list(range(2, 9))
    cv_scores = []
    for d in depths:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        score = cross_val_score(dt, Xreg, yreg, cv=5, scoring="r2").mean()
        cv_scores.append(score)

    depth_df = pd.DataFrame({"max_depth": depths, "CV R²": cv_scores})
    best_depth = depth_df.loc[depth_df["CV R²"].idxmax(), "max_depth"]
    st.dataframe(depth_df.style.background_gradient(cmap="YlOrBr"), height=200)
    st.caption(f"Best depth = **{int(best_depth)}** via 5-fold CV (prevents over-fitting).")

    tuned_tree = DecisionTreeRegressor(max_depth=int(best_depth), random_state=42)
    base_models[f"DecisionTree (depth={int(best_depth)})"] = tuned_tree

    # ---------- fit & evaluate ----------
    results, preds = {}, {}
    for name, model in base_models.items():
        model.fit(Xreg, yreg)
        pred = model.predict(Xreg)
        preds[name] = pred
        results[name] = {
            "R²"  : model.score(Xreg, yreg),
            "RMSE": np.sqrt(((pred - yreg) ** 2).mean()),
        }

    st.markdown("#### Model Metrics")
    st.dataframe(pd.DataFrame(results).T.style.background_gradient(cmap="YlOrBr"))

    # ---------- choose best on R² ----------
    best_model_name = max(results, key=lambda k: results[k]["R²"])
    best_pred = preds[best_model_name]
    st.markdown(f"##### Best model: **{best_model_name}**")

    # ---------- Actual vs Predicted scatter ----------
    fig = px.scatter(
        x=yreg,
        y=best_pred,
        labels={"x": "Actual", "y": "Predicted"},
        title=f"Actual vs Predicted — {best_model_name}",
        color_discrete_sequence=["#bca43a"],
    )
    fig.add_shape(type="line", x0=yreg.min(), x1=yreg.max(),
                  y0=yreg.min(), y1=yreg.max(),
                  line=dict(dash="dash", color="#7e7309"))
    st.plotly_chart(fig, use_container_width=True)

    # ---------- Residual plot ----------
    fig, ax = plt.subplots()
    sns.residplot(x=best_pred, y=yreg - best_pred, lowess=True,
                  color="#bca43a", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    st.pyplot(fig)
    st.caption(
        "Residual plot reveals spread & bias; the tuned tree balances variance "
        "better than an un-pruned tree would."
    )
# =======================================================================
