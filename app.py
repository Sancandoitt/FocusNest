
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from mlxtend.frequent_patterns import apriori, association_rules
from utils import load_data, get_numeric_df

st.set_page_config(page_title="FocusNest Dashboard", page_icon="🪺", layout="wide")
st.sidebar.image("assets/FocusNest_logo.png", width=180)
st.sidebar.title("FocusNest")
st.sidebar.write("Build Better Habits. Break the Social Cycle.")

DATA_PATH = "data/Dataset_for_Business_AssociationRuleReady.xlsx"
df = load_data(DATA_PATH)


# ---------- Global sidebar filters ----------
age_min, age_max = st.sidebar.slider("Age range", 18, 60, (18, 60))

gender_filter = st.sidebar.multiselect(
    "Gender",
    options=df["Gender"].unique().tolist(),
    default=df["Gender"].unique().tolist()
)

# df_view will be the filtered dataframe we use everywhere
df_view = df.query(
    "Age >= @age_min and Age <= @age_max and Gender in @gender_filter"
)

tabs = st.tabs(["Visualization","Classification","Clustering","Assoc Rules","Regression"])

# Visualization
with tabs[0]:
    st.header("Data Visualization")
    sns.set_style("whitegrid")

    # ----- KPI cards -----
    kpi1, kpi2, kpi3 = st.columns(3)

    avg_minutes = df_view["Daily_Minutes_Spent"].mean()
    heavy_pct   = (df_view["Daily_Minutes_Spent"] > 180).mean() * 100
    avg_income  = df_view["Monthly_Income"].mean()

    kpi1.metric("Avg Daily Minutes",          f"{avg_minutes:,.1f}")
    kpi2.metric("% Heavy Users (>180 min)",   f"{heavy_pct:.1f}%")
    kpi3.metric("Avg Monthly Income",         f"${avg_income:,.0f}")

    # ----- Two main plots -----
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Age")
        fig = px.histogram(
            df_view,
            x="Age",
            nbins=15,
            color_discrete_sequence=["#bca43a"]
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Minutes Spent")
        fig, ax = plt.subplots()
        sns.kdeplot(df_view["Daily_Minutes_Spent"], ax=ax, shade=True)
        st.pyplot(fig)

    # ----- Data preview -----
    st.dataframe(df_view.head())

    # ----- Income vs Minutes density -----
    st.markdown("#### Income vs Minutes Spent Density")
    fig = px.density_heatmap(
        df_view,
        x="Daily_Minutes_Spent",
        y="Monthly_Income",
        nbinsx=30,
        nbinsy=30,
        color_continuous_scale="YlOrBr"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Heavier social-media users cluster between **50–250 minutes/day** "
        "and **≤ $2 000 income**, with a small high-income heavy-use pocket."
    )
    # ----- Pay vs Willingness box-plot -----
st.markdown("#### How much do people say they'll pay?")

fig = px.box(
    df_view,
    x="Willingness_to_Subscribe",
    y="Pay_Amount",
    color="Willingness_to_Subscribe",
    color_discrete_sequence=["#f9d278", "#d6b55b", "#bca43a"]
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Median willingness to pay rises from **$0** (No) → **$2–5** (Maybe) → "
    "**$10+** (Yes).  \n"
    "↪ Use this to design freemium vs premium tiers."
)
# ----- Platform usage treemap (clean counts) -----
st.markdown("#### Where do our users spend their social time?")

platform_cols = [c for c in df.columns if c.startswith("Uses_")]

# 1️⃣ COUNT how many users tick each platform
plat_counts = (
    df_view[platform_cols]
    .sum()                                          # sums the 1-values
    .rename(lambda c: c.replace("Uses_", ""))       # tidy names
    .reset_index(name="UserCount")                  # into a dataframe
    .rename(columns={"index": "Platform"})
)

# 2️⃣ PLOT the treemap – area shows user share, colour shows count
fig = px.treemap(
    plat_counts,
    path=["Platform"],
    values="UserCount",
    color="UserCount",
    color_continuous_scale="YlOrBr",
    title="Platform share among current filter"
)
fig.update_traces(textinfo="label+percent entry")   # show % on each block
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Treemap recalculates live with the sidebar filters. "
    "It reveals which platforms deliver the biggest audience slice "
    "so FocusNest can prioritise acquisition spend."
)

# ----- Correlation heat-map -----
st.markdown("#### Numeric feature correlations")

numeric_cols = df_view.select_dtypes("number").columns
corr = df_view[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(corr, annot=False, cmap="YlOrBr", ax=ax)
st.pyplot(fig)
st.caption(
    "**Dark cells** reveal strong relationships – e.g., "
    "Daily Minutes ↔ Challenge scores.  "
    "These are candidate variables for targeting models."
)

# Classification
with tabs[1]:
    st.header("Classification")
    X = get_numeric_df(df)
    y = df['Willingness_to_Subscribe'].map({'No':0,'Maybe':1,'Yes':2})
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, stratify=y, random_state=42)
    models = {"KNN":KNeighborsClassifier(),
              "DT":DecisionTreeClassifier(),
              "RF":RandomForestClassifier(),
              "GBRT":GradientBoostingClassifier()}
    results={}
    for name,model in models.items():
        model.fit(X_train,y_train)
        pred=model.predict(X_test)
        results[name]=[accuracy_score(y_test,pred),
                       precision_score(y_test,pred,average='macro'),
                       recall_score(y_test,pred,average='macro'),
                       f1_score(y_test,pred,average='macro')]
    res_df=pd.DataFrame(results,index=['Acc','Prec','Rec','F1']).T
    st.dataframe(res_df)
    sel=st.selectbox("Confusion Matrix",list(models.keys()))
    if st.button("Show CM"):
        cm=confusion_matrix(y_test,models[sel].predict(X_test))
        st.write(cm)

# Clustering
with tabs[2]:
    st.header("Clustering")
    k = st.slider("k", 2, 10, 4)
    km = KMeans(n_clusters=k, random_state=42).fit(get_numeric_df(df))
    df["Cluster"] = km.labels_
    st.write(df.groupby("Cluster").mean(numeric_only=True))

    fig,ax=plt.subplots()
    sns.scatterplot(x='Daily_Minutes_Spent',y='Monthly_Income',hue='Cluster',data=df,ax=ax)
    st.pyplot(fig)

# Association Rules
with tabs[3]:
    st.header("Association Rules")
    bin_cols=[c for c in df.columns if set(df[c].unique())<= {0,1}]
    cols=st.multiselect("Columns",bin_cols,default=bin_cols[:2])
    sup=st.slider("min_support",0.01,0.2,0.05)
    if cols:
        rules=association_rules(apriori(df[cols],min_support=sup,use_colnames=True),metric="confidence",min_threshold=0.3)
        st.dataframe(rules[['antecedents','consequents','support','confidence','lift']].head(10))

# Regression
with tabs[4]:
    st.header("Regression")
    target='Pay_Amount'
    X=get_numeric_df(df).drop(columns=[target])
    y=df[target]
    models={'Linear':LinearRegression(),'Ridge':Ridge(),'Lasso':Lasso()}
    for name,model in models.items():
        model.fit(X,y)
        st.write(f"{name} R2:", model.score(X,y))
