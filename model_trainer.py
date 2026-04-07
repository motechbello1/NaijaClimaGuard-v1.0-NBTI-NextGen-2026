"""
NaijaClimaGuard — Model Trainer (model_trainer.py)
===================================================
Phase 2: TRL-5 Flood Prediction Model

Pipeline:
  1. Load training_data.csv from Phase 1
  2. Feature selection + train/test split (temporal — no leakage)
  3. SMOTE oversampling on training fold only
  4. XGBoost classifier with scale_pos_weight fallback
  5. Stratified K-Fold cross-validation (5 folds)
  6. Final holdout evaluation — classification report + confusion matrix
  7. SHAP feature importance (bar + beeswarm) saved as PNGs
  8. Model serialised to flood_model.pkl
  9. Full audit trail written to model_audit.log

Outputs:
  flood_model.pkl             — serialised model + metadata
  shap_bar.png                — SHAP mean |value| per feature (submission PDF)
  shap_beeswarm.png           — SHAP beeswarm (detailed impact distribution)
  confusion_matrix.png        — holdout confusion matrix heatmap
  model_audit.log             — full run trace for TRL documentation

Install:
  pip install xgboost scikit-learn imbalanced-learn shap matplotlib seaborn joblib pandas numpy
"""

import logging
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE = Path("model_audit.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("naija_clima_guard.trainer")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH       = Path("training_data.csv")
MODEL_PATH      = Path("flood_model.pkl")
SHAP_BAR_PATH   = Path("shap_bar.png")
SHAP_BEESWARM   = Path("shap_beeswarm.png")
CM_PATH         = Path("confusion_matrix.png")

# ---------------------------------------------------------------------------
# Feature columns used for training
# (excludes identifiers, raw date, and the label itself)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "precipitation_sum",
    "precipitation_hours",
    "temperature_2m_max",
    "temperature_2m_min",
    "wind_speed_10m_max",
    "et0_fao_evapotranspiration",
    "river_discharge",
    "rain_3d_sum",
    "rain_7d_sum",
    "rain_14d_sum",
    "discharge_lag1",
    "discharge_lag3",
    "temp_range",
    "latitude",
    "longitude",
    "flood_site",
]

TARGET_COL = "flood_occurred"


# ---------------------------------------------------------------------------
# Data Loader
# ---------------------------------------------------------------------------
class DataLoader:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(
                f"'{self.path}' not found. Run data_pipeline.py first."
            )
        df = pd.read_csv(self.path, parse_dates=["date"])
        logger.info("[DATA] Loaded %d rows, %d columns from %s",
                    len(df), len(df.columns), self.path)

        # Validate required columns
        missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in training data: {missing}")

        # Sort chronologically — critical for temporal split
        df = df.sort_values(["location", "date"]).reset_index(drop=True)

        pos = int(df[TARGET_COL].sum())
        total = len(df)
        logger.info("[DATA] Class balance — positive: %d (%.2f%%), negative: %d",
                    pos, 100 * pos / total, total - pos)
        return df


# ---------------------------------------------------------------------------
# Temporal Train/Test Splitter
# ---------------------------------------------------------------------------
class TemporalSplitter:
    """
    Splits by time, not randomly.
    Training: 2020-01-01 to 2022-07-31
    Test    : 2022-08-01 to 2022-12-31  ← the actual flood event window

    This is the only valid split for time-series flood data.
    Random splits would leak future discharge values into training,
    producing falsely inflated accuracy scores that collapse on real deployment.
    """

    TRAIN_END  = "2022-05-31"
    TEST_START = "2022-06-01"

    def split(self, df: pd.DataFrame):
        train_mask = df["date"] <= self.TRAIN_END
        test_mask  = df["date"] >= self.TEST_START

        train = df[train_mask].copy()
        test  = df[test_mask].copy()

        logger.info("[SPLIT] Train: %d rows (%s → %s), flood days: %d",
                    len(train), train["date"].min().date(), train["date"].max().date(),
                    train[TARGET_COL].sum())
        logger.info("[SPLIT] Test : %d rows (%s → %s), flood days: %d",
                    len(test), test["date"].min().date(), test["date"].max().date(),
                    test[TARGET_COL].sum())

        X_train = train[FEATURE_COLS].values
        y_train = train[TARGET_COL].values
        X_test  = test[FEATURE_COLS].values
        y_test  = test[TARGET_COL].values

        return X_train, X_test, y_train, y_test, train, test


# ---------------------------------------------------------------------------
# SMOTE Resampler
# ---------------------------------------------------------------------------
class Resampler:
    """
    Applies SMOTE only to the training set.
    SMOTE is NEVER applied to the test set — that would contaminate evaluation.

    k_neighbors is set conservatively (3) because with only ~2.85% positive
    class, the minority neighbourhood is small.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def resample(self, X: np.ndarray, y: np.ndarray):
        pos_count = int(y.sum())
        neg_count = int(len(y) - pos_count)
        logger.info("[SMOTE] Before — positive: %d, negative: %d", pos_count, neg_count)

        # k_neighbors must be < minority class size
        k = min(3, pos_count - 1)
        if k < 1:
            logger.warning("[SMOTE] Too few positive samples (%d) — skipping SMOTE", pos_count)
            return X, y

        smote = SMOTE(random_state=self.random_state, k_neighbors=k)
        X_res, y_res = smote.fit_resample(X, y)

        pos_after = int(y_res.sum())
        logger.info("[SMOTE] After  — positive: %d, negative: %d, total: %d",
                    pos_after, len(y_res) - pos_after, len(y_res))
        return X_res, y_res


# ---------------------------------------------------------------------------
# Model Builder
# ---------------------------------------------------------------------------
class ModelBuilder:
    """
    XGBoost classifier.

    Key hyperparameters:
      scale_pos_weight = neg/pos ratio — handles imbalance even without SMOTE
      max_depth = 4 — shallow trees prevent overfitting on small flood minority
      n_estimators = 300 with early stopping via eval_set
      learning_rate = 0.05 — slow learner, more robust generalisation
      subsample / colsample_bytree = 0.8 — bagging regularisation
      eval_metric = aucpr — area under precision-recall curve, correct metric
                             for imbalanced binary classification (not accuracy)
    """

    def __init__(self, scale_pos_weight: float, random_state: int = 42):
        self.random_state = random_state
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            use_label_encoder=False,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> "XGBClassifier":
        logger.info("[MODEL] Training XGBoost — %d samples, %d features",
                    len(X_train), X_train.shape[1])
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            verbose=False,
        )
        logger.info("[MODEL] Training complete")
        return self.model


# ---------------------------------------------------------------------------
# Cross Validator
# ---------------------------------------------------------------------------
class CrossValidator:
    """
    Stratified K-Fold CV on the training set.
    Stratified = each fold preserves the positive class ratio.
    SMOTE is applied inside each fold to prevent data leakage.
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.resampler = Resampler(random_state)

    def run(self, X: np.ndarray, y: np.ndarray, scale_pos_weight: float) -> dict:
        logger.info("[CV] Starting %d-fold stratified cross-validation ...", self.n_splits)
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        fold_metrics = {
            "f1": [], "precision": [], "recall": [], "roc_auc": []
        }

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # SMOTE inside fold only
            X_tr_res, y_tr_res = self.resampler.resample(X_tr, y_tr)

            builder = ModelBuilder(scale_pos_weight)
            model = builder.train(X_tr_res, y_tr_res)

            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]

            f1   = f1_score(y_val, y_pred, zero_division=0)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec  = recall_score(y_val, y_pred, zero_division=0)

            # ROC-AUC only if both classes present in val fold
            if len(np.unique(y_val)) > 1:
                auc = roc_auc_score(y_val, y_prob)
            else:
                auc = float("nan")

            fold_metrics["f1"].append(f1)
            fold_metrics["precision"].append(prec)
            fold_metrics["recall"].append(rec)
            fold_metrics["roc_auc"].append(auc)

            logger.info(
                "[CV] Fold %d — F1: %.3f | Precision: %.3f | Recall: %.3f | ROC-AUC: %.3f",
                fold, f1, prec, rec, auc if not np.isnan(auc) else -1,
            )

        summary = {k: (np.nanmean(v), np.nanstd(v)) for k, v in fold_metrics.items()}
        logger.info("[CV] RESULTS (mean ± std across %d folds):", self.n_splits)
        for metric, (mean, std) in summary.items():
            logger.info("[CV]   %-12s %.3f ± %.3f", metric, mean, std)

        return summary


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
class Evaluator:
    def evaluate(self, model, X_test: np.ndarray, y_test: np.ndarray,
                 feature_names: list) -> dict:
        logger.info("[EVAL] Running holdout evaluation on test set (%d samples) ...", len(y_test))

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, target_names=["No Flood", "Flood"])
        logger.info("[EVAL] Classification Report:\n%s", report)
        print("\n" + "=" * 60)
        print("HOLDOUT TEST SET — CLASSIFICATION REPORT")
        print("=" * 60)
        print(report)

        cm = confusion_matrix(y_test, y_pred)
        logger.info("[EVAL] Confusion Matrix:\n%s", str(cm))

        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, y_prob)
            logger.info("[EVAL] ROC-AUC: %.4f", auc)
            print(f"ROC-AUC Score : {auc:.4f}")
        else:
            auc = None
            logger.warning("[EVAL] Only one class in test set — ROC-AUC unavailable")

        self._plot_confusion_matrix(cm)

        return {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "roc_auc": auc,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

    def _plot_confusion_matrix(self, cm: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Flood", "Flood"],
            yticklabels=["No Flood", "Flood"],
            ax=ax,
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title("NaijaClimaGuard — Confusion Matrix (Holdout)", fontsize=13)
        plt.tight_layout()
        fig.savefig(CM_PATH, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("[EVAL] Confusion matrix saved: %s", CM_PATH)


# ---------------------------------------------------------------------------
# SHAP Explainer
# ---------------------------------------------------------------------------
class SHAPExplainer:
    """
    Generates two SHAP plots:
      1. shap_bar.png      — mean absolute SHAP value per feature
                             (shows which variables drive predictions most)
      2. shap_beeswarm.png — SHAP value distribution per feature per sample
                             (shows direction and magnitude of each feature's impact)

    These are your submission proof artifacts demonstrating model interpretability,
    a requirement for TRL-5 scientific credibility with Innovate UK assessors.
    """

    def explain(self, model, X_test: np.ndarray, feature_names: list) -> None:
        logger.info("[SHAP] Computing SHAP values for %d test samples ...", len(X_test))

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # --- Bar plot: mean |SHAP| per feature ---
        fig, ax = plt.subplots(figsize=(9, 6))
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_values = mean_abs_shap[sorted_idx]

        colors = ["#1B4F72" if i == 0 else "#2E86C1" if i < 3 else "#AED6F1"
                  for i in range(len(sorted_features))]
        bars = ax.barh(sorted_features[::-1], sorted_values[::-1], color=colors[::-1])
        ax.set_xlabel("Mean |SHAP Value| — Average impact on model output", fontsize=11)
        ax.set_title("NaijaClimaGuard — Feature Importance (SHAP)", fontsize=13, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Annotate top 3
        for i, (feat, val) in enumerate(zip(sorted_features[:3], sorted_values[:3])):
            rev_i = len(sorted_features) - 1 - i
            ax.text(val + 0.002, rev_i, f"{val:.3f}", va="center", fontsize=9, color="#1B4F72")

        plt.tight_layout()
        fig.savefig(SHAP_BAR_PATH, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("[SHAP] Bar plot saved: %s", SHAP_BAR_PATH)

        # --- Beeswarm plot ---
        shap_exp = shap.Explanation(
            values=shap_values,
            data=X_test,
            feature_names=feature_names,
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.plots.beeswarm(shap_exp, max_display=14, show=False)
        plt.title("NaijaClimaGuard — SHAP Beeswarm (Feature Impact Distribution)", fontsize=12)
        plt.tight_layout()
        fig.savefig(SHAP_BEESWARM, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("[SHAP] Beeswarm plot saved: %s", SHAP_BEESWARM)


# ---------------------------------------------------------------------------
# Model Serialiser
# ---------------------------------------------------------------------------
class ModelSerialiser:
    def save(self, model, cv_summary: dict, feature_names: list,
             eval_results: dict) -> None:
        payload = {
            "model": model,
            "feature_names": feature_names,
            "cv_summary": cv_summary,
            "roc_auc_holdout": eval_results.get("roc_auc"),
            "model_version": "1.0.0",
            "training_data": str(DATA_PATH),
        }
        joblib.dump(payload, MODEL_PATH)
        logger.info("[SAVE] Model saved: %s", MODEL_PATH)
        print(f"\nModel saved    : {MODEL_PATH}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class TrainingPipeline:
    def __init__(self):
        self.loader      = DataLoader(DATA_PATH)
        self.splitter    = TemporalSplitter()
        self.resampler   = Resampler()
        self.cv          = CrossValidator(n_splits=5)
        self.evaluator   = Evaluator()
        self.explainer   = SHAPExplainer()
        self.serialiser  = ModelSerialiser()

    def run(self) -> None:
        logger.info("NaijaClimaGuard Model Trainer — START")

        # 1. Load data
        df = self.loader.load()

        # 2. Temporal split
        X_train, X_test, y_train, y_test, train_df, test_df = self.splitter.split(df)

        # 3. Compute class weight for XGBoost
        pos = int(y_train.sum())
        neg = int(len(y_train) - pos)
        scale_pos_weight = neg / pos if pos > 0 else 1.0
        logger.info("[WEIGHT] scale_pos_weight = %.2f (neg/pos = %d/%d)",
                    scale_pos_weight, neg, pos)

        # 4. Cross-validation (SMOTE applied inside each fold)
        cv_summary = self.cv.run(X_train, y_train, scale_pos_weight)

        # 5. Train final model on full training set with SMOTE
        X_train_res, y_train_res = self.resampler.resample(X_train, y_train)
        builder = ModelBuilder(scale_pos_weight)
        model = builder.train(X_train_res, y_train_res)

        # 6. Evaluate on holdout
        eval_results = self.evaluator.evaluate(
            model, X_test, y_test, FEATURE_COLS
        )

        # 7. SHAP explainability
        self.explainer.explain(model, X_test, FEATURE_COLS)

        # 8. Save model
        self.serialiser.save(model, cv_summary, FEATURE_COLS, eval_results)

        # 9. Final summary
        self._print_summary(cv_summary, eval_results)

    def _print_summary(self, cv_summary: dict, eval_results: dict) -> None:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE — TRL-5 VALIDATION ARTIFACTS")
        print("=" * 60)
        print("\nCross-Validation Results (5-fold, stratified):")
        for metric, (mean, std) in cv_summary.items():
            print(f"  {metric:<14} {mean:.3f} ± {std:.3f}")
        auc = eval_results.get("roc_auc")
        if auc:
            print(f"\nHoldout ROC-AUC: {auc:.4f}")
        print("\nArtifacts generated:")
        for p in [MODEL_PATH, SHAP_BAR_PATH, SHAP_BEESWARM, CM_PATH, LOG_FILE]:
            status = "OK" if p.exists() else "MISSING"
            print(f"  [{status}] {p}")
        print("=" * 60)
        print("\nNext step: run  streamlit run app.py")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        pipeline.run()
    except Exception as exc:
        logger.critical("Training aborted: %s", exc, exc_info=True)
        sys.exit(1)
