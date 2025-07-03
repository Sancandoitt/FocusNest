import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from mlxtend.frequent_patterns import apriori, association_rules
import streamlit as st
import re                       # ← keep just this one

# ----------------------------------------------------------------------
# Helper: tidy all column names
# ----------------------------------------------------------------------
def tidy_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    lower-cases headers, replaces punctuation/space with underscore,
    and strips leading/trailing underscores
    """
    df = df.copy()
    df.columns = [
        re.sub(r"[^0-9a-zA-Z]+", "_", col).strip("_").lower()
        for col in df.columns
    ]
    return df

# ----------------------------------------------------------------------
# Cached loader
# ----------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = tidy_column_names(df)        # clean headers
    return df

# ----------------------------------------------------------------------
# Numeric-column selector
# ----------------------------------------------------------------------
def get_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return every numeric column—no hard-coded list."""
    return df.select_dtypes("number")

