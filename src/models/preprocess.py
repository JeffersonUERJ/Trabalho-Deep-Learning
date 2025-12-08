# preprocess.py
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().strip('"') for c in df.columns]
    return df


def _month_to_int(val):
    try:
        if pd.isna(val):
            return np.nan
        s = str(val)
        m = re.search(r"(\d{1,2})", s)
        if m:
            return int(m.group(1))
        return np.nan
    except Exception:
        return np.nan


def remove_extreme_outliers(df, col, quantile=0.995):
    if col not in df.columns:
        return df
    lim = df[col].quantile(quantile)
    return df[df[col] <= lim]



def prepare_dataset(parquet_path: str,
                    test_size: float = 0.15,
                    val_size: float = 0.15,
                    random_state: int = 42,
                    target_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                             np.ndarray, np.ndarray, np.ndarray,
                                                             StandardScaler, List[str], pd.DataFrame]:
    """
    Load parquet, clean, encode, split and scale X. Returns:
    X_train_s, X_val_s, X_test_s,
    y_train_values, y_val_values, y_test_values,
    scaler_X, feature_names, df (original cleaned dataframe)
    """

    if target_cols is None:
        target_cols = ["Toneladas", "Valor US$ FOB"]

    df = pd.read_parquet(parquet_path)
    df = _clean_column_names(df)
   

    # numeric coercion
    for col in ["Valor US$ FOB"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = remove_extreme_outliers(df, "Valopr US$ FOB", quantile=0.995)

    # drop rows with missing targets (only remove rows missing any existing target)
    for t in target_cols:
        if t in df.columns:
            df = df[~df[t].isna()]

    # temporal features
    if "Ano" in df.columns:
        df["Ano"] = pd.to_numeric(df["Ano"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
    if "Mês" in df.columns:
        df["Mes_num"] = df["Mês"].apply(_month_to_int)

    # select numeric features (exclude targets)
    exclude = set([c for c in target_cols if c in df.columns])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    # small categorical set
    categorical_candidates = [c for c in ["Países", "UF do Produto", "Código NCM", "URF"] if c in df.columns]

    X = df[numeric_cols].copy()

    if categorical_candidates:
        cats = pd.get_dummies(df[categorical_candidates].astype(str).fillna("NA"), drop_first=True)
        X = pd.concat([X.reset_index(drop=True), cats.reset_index(drop=True)], axis=1)

    y_cols = [c for c in target_cols if c in df.columns]
    y = df[y_cols].copy()

    # replace inf and align
    X = X.replace([np.inf, -np.inf], np.nan)
    if not y.empty:
        mask = X.notna().all(axis=1) & (~y.isna().any(axis=1))
    else:
        mask = X.notna().all(axis=1)
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    # split
    total_test = test_size + val_size
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=total_test, random_state=random_state)
    rel_val = val_size / total_test if total_test > 0 else 0
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1 - rel_val), random_state=random_state)

    # scale X (fit on train only)
    scaler = StandardScaler()
    if X_train.shape[1] > 0:
        scaler.fit(X_train.values)
        X_train_s = scaler.transform(X_train.values)
        X_val_s = scaler.transform(X_val.values)
        X_test_s = scaler.transform(X_test.values)
    else:
        X_train_s = X_train.values
        X_val_s = X_val.values
        X_test_s = X_test.values

    return (X_train_s, X_val_s, X_test_s,
            y_train.values, y_val.values, y_test.values,
            scaler, X.columns.tolist(), df)
