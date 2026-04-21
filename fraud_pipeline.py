"""
fraud_pipeline.py
=================
Multi-file fraud detection pipeline for all 4 CSV files:

  FILE                              SCHEMA
  ──────────────────────────────────────────────────────
  fraudTrain.csv                    Transactions (Kaggle)
  fraudTest.csv                     Transactions (Kaggle)
  df_2026-03-10 05_53_16.csv        Transactions (sample)
  creditcard.csv                    European PCA (V1-V28)

Auto-detects schema, standardises both into a UNIFIED feature set,
tags each row with its source, merges, deduplicates, and splits
into train/test for ML.

University of the West of Scotland — MSc Project
Evans Polley | B01823633
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from sklearn.impute import SimpleImputer

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# ──────────────────────────────────────────────────────────────
#  UNIFIED FEATURE COLUMNS  (both schemas map into this space)
# ──────────────────────────────────────────────────────────────
#
#  Shared features (populated by both schemas):
#    amt, trans_hour, trans_dow, trans_month, trans_day,
#    city_pop, age, geo_distance, is_online, amt_zscore
#
#  Transactions-only (0 for creditcard rows):
#    cat_* (14 one-hot category flags)
#
#  Creditcard-only (0 for transactions rows):
#    v1 … v28  (PCA components)

FEATURE_COLS = [
    # ── Shared ──
    "amt", "trans_hour", "trans_dow", "trans_month", "trans_day",
    "city_pop", "age", "geo_distance", "is_online", "amt_zscore",
    # ── Transaction categories (OHE) ──
    "cat_food_dining", "cat_gas_transport", "cat_grocery_net", "cat_grocery_pos",
    "cat_health_fitness", "cat_home", "cat_kids_pets", "cat_misc_net",
    "cat_misc_pos", "cat_personal_care", "cat_shopping_net",
    "cat_shopping_pos", "cat_travel", "cat_entertainment",
    # ── PCA features (European dataset) ──
    "v1","v2","v3","v4","v5","v6","v7","v8","v9","v10",
    "v11","v12","v13","v14","v15","v16","v17","v18","v19","v20",
    "v21","v22","v23","v24","v25","v26","v27","v28",
]

SCHEMA_TRANSACTIONS = "transactions"
SCHEMA_CREDITCARD   = "creditcard"


# ──────────────────────────────────────────────────────────────
#  SCHEMA DETECTION
# ──────────────────────────────────────────────────────────────

def detect_schema(df: pd.DataFrame) -> str:
    cols = [c.lower().strip() for c in df.columns]
    has_v_pca = sum(1 for c in cols if c.startswith("v") and c[1:].isdigit()) >= 10
    has_class = "class" in cols
    has_merchant = "merchant" in cols or "trans_num" in cols
    if has_v_pca and has_class:
        return SCHEMA_CREDITCARD
    if has_merchant:
        return SCHEMA_TRANSACTIONS
    if has_v_pca:
        return SCHEMA_CREDITCARD
    return SCHEMA_TRANSACTIONS


# ──────────────────────────────────────────────────────────────
#  STANDARDISE: TRANSACTIONS SCHEMA
# ──────────────────────────────────────────────────────────────

CATEGORY_MAP = {
    "food_dining":    "cat_food_dining",
    "gas_transport":  "cat_gas_transport",
    "grocery_net":    "cat_grocery_net",
    "grocery_pos":    "cat_grocery_pos",
    "health_fitness": "cat_health_fitness",
    "home":           "cat_home",
    "kids_pets":      "cat_kids_pets",
    "misc_net":       "cat_misc_net",
    "misc_pos":       "cat_misc_pos",
    "personal_care":  "cat_personal_care",
    "shopping_net":   "cat_shopping_net",
    "shopping_pos":   "cat_shopping_pos",
    "travel":         "cat_travel",
    "entertainment":  "cat_entertainment",
}


def standardise_transactions(df: pd.DataFrame, source_tag: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("unnamed")]

    # Target
    for cand in ["is_fraud", "fraud", "label"]:
        if cand in df.columns:
            df.rename(columns={cand: "is_fraud"}, inplace=True); break
    if "is_fraud" not in df.columns:
        df["is_fraud"] = 0

    # Drop PII
    df.drop(columns=[c for c in ["cc_num","first","last","gender","street","zip",
                                   "unix_time","acct_num"] if c in df.columns],
            inplace=True, errors="ignore")

    # Datetime features
    df["trans_datetime"] = pd.NaT
    for col in ["trans_date_trans_time", "datetime", "timestamp"]:
        if col in df.columns:
            dt = pd.to_datetime(df[col], utc=True, errors="coerce")
            df["trans_datetime"] = dt
            df["trans_hour"]  = dt.dt.hour.fillna(0).astype(int)
            df["trans_dow"]   = dt.dt.dayofweek.fillna(0).astype(int)
            df["trans_month"] = dt.dt.month.fillna(1).astype(int)
            df["trans_day"]   = dt.dt.day.fillna(1).astype(int)
            break
    for col in ["trans_hour", "trans_dow", "trans_month", "trans_day"]:
        if col not in df.columns:
            df[col] = 0

    # Age
    if "dob" in df.columns:
        try:
            dob = pd.to_datetime(df["dob"], utc=True, errors="coerce")
            df["age"] = ((pd.Timestamp("2019-01-01", tz="UTC") - dob)
                         .dt.days / 365.25).clip(18, 100).fillna(40)
        except Exception:
            df["age"] = 40.0
    else:
        df["age"] = 40.0

    # Amount
    amt_col = "amt" if "amt" in df.columns else "amount"
    df["amt"] = pd.to_numeric(df.get(amt_col, pd.Series([0]*len(df))),
                               errors="coerce").fillna(0)

    # Geo distance
    if all(c in df.columns for c in ["lat", "long", "merch_lat", "merch_long"]):
        df["geo_distance"] = np.sqrt(
            (df["lat"].astype(float) - df["merch_lat"].astype(float))**2 +
            (df["long"].astype(float) - df["merch_long"].astype(float))**2
        ).fillna(0)
    else:
        df["geo_distance"] = 0.0

    df["city_pop"] = pd.to_numeric(df.get("city_pop", pd.Series([0]*len(df))),
                                    errors="coerce").fillna(0)
    df["is_online"] = (df.get("category", pd.Series([""] * len(df)))
                        .astype(str).str.endswith("_net").astype(int))

    std = df["amt"].std() + 1e-9
    df["amt_zscore"] = ((df["amt"] - df["amt"].mean()) / std).fillna(0)

    # Category OHE
    cat_col = df.get("category", pd.Series([""] * len(df))).astype(str).str.lower()
    for raw, col in CATEGORY_MAP.items():
        df[col] = (cat_col == raw).astype(int)

    # PCA features = 0
    for i in range(1, 29):
        df[f"v{i}"] = 0.0

    df["source_tag"] = source_tag
    df["trans_num"]  = df.get("trans_num", pd.Series([""] * len(df))).fillna("").astype(str)
    return df


# ──────────────────────────────────────────────────────────────
#  STANDARDISE: CREDITCARD (PCA) SCHEMA
# ──────────────────────────────────────────────────────────────

def standardise_creditcard(df: pd.DataFrame, source_tag: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Target
    for cand in ["class", "is_fraud", "fraud", "label"]:
        if cand in df.columns:
            df.rename(columns={cand: "is_fraud"}, inplace=True); break
    if "is_fraud" not in df.columns:
        df["is_fraud"] = 0

    # Amount
    for cand in ["amount", "amt"]:
        if cand in df.columns:
            df["amt"] = pd.to_numeric(df[cand], errors="coerce").fillna(0); break
    if "amt" not in df.columns:
        df["amt"] = 0.0

    # Time → approximate temporal features
    if "time" in df.columns:
        t = pd.to_numeric(df["time"], errors="coerce").fillna(0)
        df["trans_hour"]  = (t // 3600 % 24).astype(int)
        df["trans_dow"]   = (t // 86400 % 7).astype(int)
        df["trans_month"] = 0
        df["trans_day"]   = (t // 86400 % 30).astype(int)
    else:
        df["trans_hour"] = df["trans_dow"] = df["trans_month"] = df["trans_day"] = 0

    df["trans_datetime"] = pd.NaT
    df["age"]            = 40.0
    df["geo_distance"]   = 0.0
    df["city_pop"]       = 0
    df["is_online"]      = 0

    std = df["amt"].std() + 1e-9
    df["amt_zscore"] = ((df["amt"] - df["amt"].mean()) / std).fillna(0)

    # Category flags = 0
    for col in CATEGORY_MAP.values():
        df[col] = 0

    # PCA features
    for i in range(1, 29):
        col = f"v{i}"
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

    df["source_tag"] = source_tag
    df["trans_num"]  = ""
    return df


# ──────────────────────────────────────────────────────────────
#  PIPELINE
# ──────────────────────────────────────────────────────────────

class FraudDataPipeline:
    """
    Load one or all four CSV files, standardise, merge, split.

    Single file:
        p = FraudDataPipeline()
        df = p.load("/content/creditcard.csv")

    All four files:
        df = p.load_all([
            "/content/fraudTrain.csv",
            "/content/fraudTest.csv",
            "/content/df_2026-03-10 05_53_16.csv",
            "/content/creditcard.csv",
        ])
        p.print_report()
    """

    def __init__(self):
        self.raw_frames   = {}
        self.clean_frames = {}
        self.df           = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.scaler      = StandardScaler()
        self.load_report = []

    def load(self, filepath: str) -> pd.DataFrame:
        """Load a single file (backwards compatible)."""
        return self.load_all([filepath])

    def load_all(self, filepaths: list) -> pd.DataFrame:
        """Load, standardise, and merge all given files."""
        self.raw_frames.clear()
        self.clean_frames.clear()
        self.load_report.clear()

        for path in filepaths:
            if not os.path.exists(path):
                self.load_report.append(f"NOT FOUND: {path}")
                continue
            try:
                raw    = self._read_file(path)
                tag    = self._source_tag(path)
                schema = detect_schema(raw)
                self.raw_frames[tag] = raw

                if schema == SCHEMA_CREDITCARD:
                    clean = standardise_creditcard(raw, tag)
                else:
                    clean = standardise_transactions(raw, tag)

                self.clean_frames[tag] = clean
                nf = int(clean["is_fraud"].sum()); n = len(clean)
                self.load_report.append(
                    f"OK  {os.path.basename(path):<42} schema={schema:<14} "
                    f"rows={n:>9,}  fraud={nf:>7,} ({nf/n*100:.3f}%)"
                )
            except Exception as e:
                self.load_report.append(f"ERROR  {path}: {e}")

        self._merge()
        return self.df

    def _merge(self):
        if not self.clean_frames:
            raise ValueError("No files loaded successfully.")

        all_cols = FEATURE_COLS + ["is_fraud", "source_tag", "trans_num", "trans_datetime"]
        aligned = []
        for frame in self.clean_frames.values():
            for col in all_cols:
                if col not in frame.columns:
                    frame[col] = 0
            aligned.append(frame[all_cols].copy())

        merged = pd.concat(aligned, ignore_index=True)

        # Deduplicate on trans_num (where available)
        before = len(merged)
        with_tn    = merged[merged["trans_num"].str.len() > 0]
        without_tn = merged[merged["trans_num"].str.len() == 0]
        with_tn    = with_tn.drop_duplicates(subset=["trans_num"])
        merged     = pd.concat([with_tn, without_tn], ignore_index=True)

        # Deduplicate creditcard rows on (amt, v1, is_fraud)
        cc_mask    = merged["v1"] != 0
        cc_deduped = merged[cc_mask].drop_duplicates(
            subset=["amt", "v1", "is_fraud"])
        non_cc     = merged[~cc_mask]
        merged     = pd.concat([non_cc, cc_deduped], ignore_index=True)
        after      = len(merged)

        self.load_report.append(
            f"\nMERGED: {after:,} total rows  "
            f"({before - after:,} duplicates removed)  |  "
            f"Fraud: {int(merged['is_fraud'].sum()):,} "
            f"({merged['is_fraud'].mean()*100:.4f}%)"
        )

        # Shuffle
        merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

        # Impute NaNs in feature columns
        numeric_cols = [c for c in FEATURE_COLS
                        if merged[c].dtype in (np.float64, np.float32,
                                                np.int64, np.int32)]
        imp = SimpleImputer(strategy="median")
        merged[numeric_cols] = imp.fit_transform(merged[numeric_cols])

        self.df = merged

    def split_and_resample(self, test_size: float = 0.2,
                            use_smote: bool = True) -> tuple:
        """Stratified split → scale (train only) → optional SMOTE (train only)."""
        if self.df is None:
            raise ValueError("Load data first.")

        df = self.df.dropna(subset=["is_fraud"])
        X  = df[FEATURE_COLS].values.astype(float)
        y  = df["is_fraud"].values.astype(int)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42)

        X_tr = self.scaler.fit_transform(X_tr)   # fit on train only
        X_te = self.scaler.transform(X_te)        # transform test (no fit)

        if use_smote and SMOTE_AVAILABLE and y_tr.sum() >= 6:
            try:
                k  = min(5, y_tr.sum() - 1)
                sm = SMOTE(random_state=42, k_neighbors=k)
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
                self.load_report.append(
                    f"SMOTE  train rebalanced → "
                    f"{int(y_tr.sum()):,} fraud / {int((y_tr==0).sum()):,} legit"
                )
            except Exception as e:
                self.load_report.append(f"SMOTE skipped: {e}")

        self.X_train, self.X_test = X_tr, X_te
        self.y_train, self.y_test = y_tr, y_te
        return X_tr, X_te, y_tr, y_te

    def print_report(self):
        print("\n" + "=" * 75)
        print("  FRAUD PIPELINE — LOAD & MERGE REPORT")
        print("=" * 75)
        for line in self.load_report:
            print(" ", line)
        print("=" * 75 + "\n")

    def get_source_summary(self) -> pd.DataFrame:
        if self.df is None:
            return pd.DataFrame()
        return (self.df.groupby("source_tag")["is_fraud"]
                .agg(total="count",
                     fraud=lambda x: int((x == 1).sum()),
                     legit=lambda x: int((x == 0).sum()),
                     fraud_pct=lambda x: f"{x.mean()*100:.3f}%")
                .reset_index())

    @property
    def feature_names(self):
        return FEATURE_COLS

    @staticmethod
    def _read_file(path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[-1].lower()
        if ext == ".csv":
            return pd.read_csv(path, low_memory=False)
        elif ext in (".xlsx", ".xls"):
            return pd.read_excel(path)
        elif ext == ".json":
            return pd.read_json(path)
        raise ValueError(f"Unsupported file type: {ext}")

    @staticmethod
    def _source_tag(path: str) -> str:
        name = os.path.basename(path).lower()
        if "fraudtrain" in name:  return "fraud_train"
        if "fraudtest"  in name:  return "fraud_test"
        if "creditcard" in name:  return "creditcard_eu"
        return "sample_data"


# ──────────────────────────────────────────────────────────────
#  MODEL MANAGER
# ──────────────────────────────────────────────────────────────

class ModelManager:
    MODELS = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42),
        "Decision Tree":       DecisionTreeClassifier(
            max_depth=10, class_weight="balanced", random_state=42),
        "Random Forest":       RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1),
        "SVM":                 SVC(
            kernel="rbf", class_weight="balanced", probability=True, random_state=42),
    }

    def __init__(self):
        self.trained = {}
        self.results = {}

    def train_all(self, X_train, y_train, X_test, y_test, progress_cb=None):
        self.results = {}; self.trained = {}
        for name, model in self.MODELS.items():
            if progress_cb:
                progress_cb(f"Training {name}…")
            try:
                m = model.__class__(**model.get_params())
                m.fit(X_train, y_train)
                y_pred = m.predict(X_test)
                try:
                    auc = roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
                except Exception:
                    auc = 0.0
                self.trained[name] = m
                self.results[name] = {
                    "accuracy":  accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                    "recall":    recall_score(y_test, y_pred, zero_division=0),
                    "f1":        f1_score(y_test, y_pred, zero_division=0),
                    "auc":       auc,
                    "cm":        confusion_matrix(y_test, y_pred),
                    "y_pred":    y_pred,
                }
            except Exception as e:
                self.results[name] = {"error": str(e)}
        return self.results

    def train_single(self, name, X_train, y_train, X_test, y_test):
        m = self.MODELS[name].__class__(**self.MODELS[name].get_params())
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        try:
            auc = roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
        except Exception:
            auc = 0.0
        self.trained[name] = m
        self.results[name] = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall":    recall_score(y_test, y_pred, zero_division=0),
            "f1":        f1_score(y_test, y_pred, zero_division=0),
            "auc":       auc,
            "cm":        confusion_matrix(y_test, y_pred),
            "y_pred":    y_pred,
        }
        return self.results[name]
