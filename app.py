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

from utils import load_data, get_numeric_df, tidy_column_names

# ----------------------------------------------------------------------
# Basic page config
# ----------------------------------------------------------------------
st.set_page_config(page_title="FocusNest Dashboard",
                   page_icon="🪺", layout="wide")
st.sidebar.image("assets/FocusNest_logo.png", width=180)
st.sidebar.title("FocusNest")
st.sidebar.write("Build Better Habits. Break the Social Cycle.")

# ----------------------------------------------------------------------
# 📂 Data source – built-in sample  OR  user upload
# ---------------------------------------------------------------
DATA_PATH = "data/Dataset_for_Business_AssociationRuleReady.xlsx"   # keep your original file

uploaded = st.sidebar.file_uploader(
    "Upload CSV / Excel", type=["csv", "xlsx"],
    help="If blank, the dashboard uses the default FocusNest dataset."
)

@st.cache_data(show_spinner=False)
def read_data(file) -> pd.DataFrame:
    """Return a tidy DataFrame from either the sample file or an upload."""
    if file is None:                              # built-in sample
        df_ = load_data(DATA_PATH)
    elif file.name.endswith(".csv"):
        df_ = pd.read_csv(file)
    else:                                         # .xlsx
        df_ = pd.read_excel(file)
    return tidy_column_names(df_)                 # unify column names

df = read_data(uploaded)                          # <<< global DataFrame

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
tabs = st.tabs([
    "📊 Visualisation", "🤖 Classification", "👥 Clustering",
    "🔗 Assoc Rules", "📈 Regression", "📝 Insights"   # ← new
])

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

    # Build the correlation matrix first!
    numeric_cols = df_view.select_dtypes("number").columns
    corr = df_view[numeric_cols].corr()

    st.markdown("#### Correlation Heat-map (continuous features)")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        cmap="YlOrBr",
        annot=False,
        ax=ax,
        vmin=-1, vmax=1,
        linewidths=0.3,
        cbar_kws={'label': 'Correlation'}
    )
    st.pyplot(fig)

    mask = np.triu(np.ones_like(corr, dtype=bool))
    top5 = (
        corr.abs()
             .where(~mask)
             .stack()
             .sort_values(ascending=False)
             .head(5)
             .reset_index()
             .rename(columns={
                 "level_0": "Feature 1",
                 "level_1": "Feature 2",
                 0: "|r|"
             })
             .round(2)
    )
    st.markdown("**Top-5 strongest correlations**")
    st.dataframe(top5)


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

        # ---------- Download predictions CSV ----------
        # build a DataFrame with actual + selected-model prediction
        pred_df = X_test.copy()
        pred_df["actual"] = y_test.values
        pred_df["predicted"] = models[cm_choice].predict(X_test)
        csv = pred_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download {cm_choice} predictions (CSV)",
            data=csv,
            file_name=f"{cm_choice.lower()}_predictions.csv",
            mime="text/csv",
        )


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
      # ---------- EXTRA: one-hot models + manual What-If -------------
    st.subheader("🔍 Extended Regression")

    # 1️⃣  one-hot encode all categoricals
    X_catfree = df_view.drop(columns=[target])          # remove chosen target
    X_encoded = pd.get_dummies(X_catfree, drop_first=True)
    y_encoded = df_view[target]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_encoded, y_encoded, test_size=0.3, random_state=42
    )

    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.preprocessing import PolynomialFeatures

    ext_models = {
        "Linear":       LinearRegression(),
        "Ridge":        Ridge(),
        "Lasso":        Lasso(),
        "DecisionTree": DecisionTreeRegressor(random_state=42)
    }

    # 2️⃣  fit & score
    perf_rows = []
    for mname, m in ext_models.items():
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        perf_rows.append({
            "Model": mname,
            "R²":   round(r2_score(y_te, pred), 3),
            "RMSE": round(np.sqrt(mean_squared_error(y_te, pred)), 2)
        })
        ext_models[mname] = m   # keep fitted


    perf_df = pd.DataFrame(perf_rows).set_index("Model")
    st.dataframe(perf_df.style.background_gradient(cmap="YlOrBr"))

    # 3️⃣  Manual What-If inputs for five key variables
    key_inputs = [
        "age", "monthly_income", "daily_minutes_spent",
        "willingness", "main_challenge"
    ]
    st.markdown("#### Manual What-If (enter 5 key variables)")
    user_vals = {}
    with st.form("manual_pred_form"):
        for col in X_catfree.columns:
            if col in key_inputs:
                if str(X_catfree[col].dtype).startswith(("float", "int")):
                    user_vals[col] = st.number_input(
                        col, value=float(df_view[col].mean()), format="%.2f"
                    )
                else:
                    opts = sorted(df_view[col].dropna().unique().astype(str))
                    user_vals[col] = st.selectbox(col, opts, key=f"man_{col}")
        submit_pred = st.form_submit_button("Predict")

    if submit_pred:
        # assemble one row with user inputs + defaults
        row = {}
        for c in X_catfree.columns:
            if c in user_vals:
                row[c] = user_vals[c]
            elif str(X_catfree[c].dtype).startswith(("float", "int")):
                row[c] = float(df_view[c].mean())
            else:
                row[c] = df_view[c].mode().iloc[0]
        row_df  = pd.DataFrame([row])
        row_enc = pd.get_dummies(row_df, drop_first=True)
        row_enc = row_enc.reindex(columns=X_encoded.columns, fill_value=0)

        preds = {m: round(ext_models[m].predict(row_enc)[0], 2)
                 for m in ext_models}

        out_df = perf_df.copy()
        out_df["Manual Pred"] = pd.Series(preds)
        st.dataframe(out_df)

        # bar chart of manual predictions
        chart_df = pd.DataFrame(preds, index=["Prediction"]).T
        st.bar_chart(chart_df)

    # ---------- Scatter: Actual vs Predicted for best model ----------
    best_name = max(results, key=lambda k: results[k]["R²"])
    best_pred = preds[best_name]

    # build DataFrame, drop NaNs so x- and y-lengths match
    scatter_df = pd.DataFrame({
        "Actual":    yreg.values,
        "Predicted": best_pred
    }).dropna()

    fig = px.scatter(
        scatter_df, x="Actual", y="Predicted",
        labels={"Actual": "Actual", "Predicted": "Predicted"},
        title=f"Actual vs Predicted — {best_name}",
        color_discrete_sequence=["#bca43a"]
    )
    fig.add_shape(
        type="line",
        x0=scatter_df["Actual"].min(),  x1=scatter_df["Actual"].max(),
        y0=scatter_df["Actual"].min(),  y1=scatter_df["Actual"].max(),
        line=dict(dash="dash", color="#7e7309")
    )
    st.plotly_chart(fig, use_container_width=True)


    # ---------- Residual plot ----------
    resid_df = pd.DataFrame({
        "Predicted": best_pred,
        "Residual":  yreg.values - best_pred
    }).dropna()

    fig, ax = plt.subplots()
    sns.residplot(
        data=resid_df,
        x="Predicted", y="Residual",
        lowess=True, color="#bca43a", ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    st.pyplot(fig)

    ax.set_xlabel("Predicted"); ax.set_ylabel("Residuals")
    st.pyplot(fig)
    st.caption("Residual plot checks bias/heteroscedasticity.")

    # ---------- Download regression predictions ----------
    reg_df = pd.DataFrame({
        "actual": yreg,
        "predicted": best_pred,
    })
    csv_reg = reg_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download {best_name} predictions (CSV)",
        data=csv_reg,
        file_name=f"{best_name.lower().replace(' ', '_')}_predictions.csv",
        mime="text/csv",
    )

# =====================================================================
# 6. INSIGHTS TAB  —  auto-generated executive summary
# =====================================================================
with tabs[5]:
    st.header("Executive Insights")

    # ---------- 1. Heavy-use share ----------
    heavy_pct = (df_view["daily_minutes_spent"] > 180).mean() * 100

    # ---------- 2. Median pay per intent ----------
    pay_meds = (
        df_view.groupby("willingness_to_subscribe")["pay_amount"]
        .median().round(2)
    )  # keys may include 'no', 'maybe', 'yes'

    # ---------- 3. Top-3 platforms ----------
    plat_cols = [c for c in df.columns if c.startswith("uses_")]
    plat_counts = (df_view[plat_cols].sum()
                   .rename(lambda c: c.replace("uses_", ""))
                   .sort_values(ascending=False))
    top3 = plat_counts.head(3)

    # ---------- 4. Well-being tool adoption ----------
    blocker_pct = (df_view["tried_app_blockers"] == 1).mean() * 100 \
                  if "tried_app_blockers" in df_view else np.nan
    prodapp_pct = (df_view["tried_productivity_apps"] == 1).mean() * 100 \
                  if "tried_productivity_apps" in df_view else np.nan

    # ---------- 5. Cluster persona ----------
    if "cluster" not in df.columns:          # ensure clusters exist
        km_ins = KMeans(n_clusters=4, random_state=42).fit(get_numeric_df(df))
        df["cluster"] = km_ins.labels_
    cluster_counts = df["cluster"].value_counts()
    top_cluster = int(cluster_counts.idxmax())
    cluster_profile = (df.groupby("cluster")
                         [["daily_minutes_spent", "monthly_income"]]
                         .mean().loc[top_cluster])

    # ---------- 6. Strongest numeric correlation ----------
    corr = df_view.select_dtypes("number").corr().abs()
    np.fill_diagonal(corr.values, 0)
    max_pair = corr.stack().idxmax()
    max_val  = corr.stack().max()

    # ---------- 7. Best regression R² ----------
    best_r2 = None
    best_target = None
    if "results" in globals():                      # from Regression tab
        best_target = st.session_state.get("reg_target", "pay_amount")
        best_r2 = max(results.values(), key=lambda x: x["R²"])["R²"]

    # ---------- Markdown summary ----------
    st.markdown(f"""
### Key Findings (live data)

* **Heavy-use pocket:** **{heavy_pct:.1f} %** spend > 180 min/day.
* **Median pay (USD):** No = ${pay_meds.get('no',0)}, Maybe = ${pay_meds.get('maybe',0)}, Yes = ${pay_meds.get('yes',0)}.
* **Top platforms:** {', '.join(f"{n.title()} ({c})" for n, c in top3.items())}.
* **Well-being adoption:** {blocker_pct:.1f} % tried app-blockers; {prodapp_pct:.1f} % tried productivity apps.
* **Largest persona – Cluster {top_cluster}:** avg minutes {cluster_profile['daily_minutes_spent']:.0f}, income ${cluster_profile['monthly_income']:.0f} (size {cluster_counts.max()}).
* **Strongest correlation:** {max_pair[0]} ↔ {max_pair[1]} (|r| = {max_val:.2f}).
* **Best regression R²:** {best_r2:.3f} on target **{best_target}** *(see Regression tab)*.
""")

    # ---------- Next-steps bullet list ----------
    if pay_meds.get('yes', 0) > 0:
        price_line = (
            f"Launch premium upsell at **\\${pay_meds['yes']}** to the \"Yes\" cohort."
        )
    else:
        price_line = (
            "Run price-sensitivity test (e.g. \\$2, \\$5) on the \"Yes\" cohort."
        )

    st.markdown(f"""
    ### Next Steps
    1. {price_line}
    2. Focus acquisition on **{top3.index[0].title()} / {top3.index[1].title()}** – they cover most of the audience.
    3. Build habit-formation module for Cluster {top_cluster} (heavy use, lower income).
    4. Incorporate content addressing the *{max_pair[1].replace('_',' ')}* driver identified.
    """)
