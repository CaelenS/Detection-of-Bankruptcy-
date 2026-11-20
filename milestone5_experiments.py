
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


SEED = 42
HORIZONS = ["1year", "2year", "3year", "4year", "5year"]
BOOTSTRAP_SAMPLES = 1000

# Set of ratios that can be calculated from most public companies.  
PUBLIC_FINANCIAL_FEATURES = [
    "net profit / total assets",
    "total liabilities / total assets",
    "working capital / total assets",
    "current assets / short-term liabilities",
    "retained earnings / total assets",
    "EBIT / total assets",
    "sales / total assets",
    "equity / total assets",
    "gross profit / sales",
    "logarithm of total assets",
]
SELECTION_METRIC = "pr_auc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path, default=Path.cwd(),
        help="Repository containing data/cleaned (default: current folder).",
    )
    return parser.parse_args()


def metric_row(y_true: pd.Series, predictions: np.ndarray, probabilities: np.ndarray) -> dict:
    """Return threshold and probability metrics for the bankrupt (positive) class."""
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "bankrupt_found": int(tp),
    }


def make_model(model_name: str, params: dict):
    if model_name == "Majority baseline":
        return DummyClassifier(strategy="most_frequent")
    if model_name == "Logistic Regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=SEED, **params),
        )
    if model_name == "Class-weighted Decision Tree":
        return DecisionTreeClassifier(
            class_weight="balanced", random_state=SEED, **params
        )
    if model_name == "Random Forest":
        return RandomForestClassifier(
            class_weight="balanced", random_state=SEED, n_jobs=-1, **params
        )
    raise ValueError(f"Unknown model: {model_name}")


MODEL_GRIDS = {
    "Majority baseline": [{}],
    # Four modest alternatives for each tunable model. Models are selected by
    # validation PR-AUC and recall and false positives are reported separately.
    "Logistic Regression": [
        {"C": 0.1, "class_weight": None},
        {"C": 1.0, "class_weight": None},
        {"C": 0.1, "class_weight": "balanced"},
        {"C": 1.0, "class_weight": "balanced"},
    ],
    "Class-weighted Decision Tree": [
        {"max_depth": 3, "min_samples_leaf": 1},
        {"max_depth": 5, "min_samples_leaf": 1},
        {"max_depth": 7, "min_samples_leaf": 1},
        {"max_depth": 5, "min_samples_leaf": 5},
    ],
    "Random Forest": [
        {"n_estimators": 100, "max_depth": 5, "min_samples_leaf": 1},
        {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 1},
        {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 1},
        {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 1},
    ],
}


def tune_one_model(model_name, X_train, y_train, X_validation, y_validation, horizon, log_rows):
    candidates = []
    for number, params in enumerate(MODEL_GRIDS[model_name], start=1):
        started = time.perf_counter()
        model = make_model(model_name, params)
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_validation)[:, 1]
        predictions = model.predict(X_validation)
        scores = metric_row(y_validation, predictions, probabilities)
        elapsed = time.perf_counter() - started
        candidate = {"params": params, "scores": scores, "model": model}
        candidates.append(candidate)
        log_rows.append({
            "run_id": f"{horizon}-{model_name.lower().replace(' ', '-')}-val-{number}",
            "stage": "validation_tuning",
            "horizon": horizon,
            "model": model_name,
            "parameters": json.dumps(params, sort_keys=True),
            "selection_metric": SELECTION_METRIC,
            "seed": SEED,
            "split_used": "training + validation; test not used",
            "accuracy": scores["accuracy"],
            "precision": scores["precision"],
            "recall": scores["recall"],
            "f1": scores["f1"],
            "roc_auc": scores["roc_auc"],
            "pr_auc": scores["pr_auc"],
            "false_positives": scores["false_positives"],
            "false_negatives": scores["false_negatives"],
            "runtime_seconds": round(elapsed, 2),
            "status": "completed",
            "notes": "Candidate retained only if it has the strongest validation PR-AUC.",
        })
    # Deterministic ties: favor lower false positives, then the first config.
    return max(candidates, key=lambda item: (item["scores"][SELECTION_METRIC], -item["scores"]["false_positives"]))


def bootstrap_intervals(y_true, probabilities, rng):
    """Percentile 95% CIs from stratified bootstrap test resamples."""
    y = np.asarray(y_true)
    p = np.asarray(probabilities)
    negative_indices = np.flatnonzero(y == 0)
    positive_indices = np.flatnonzero(y == 1)
    records = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = np.concatenate([
            rng.choice(negative_indices, size=len(negative_indices), replace=True),
            rng.choice(positive_indices, size=len(positive_indices), replace=True),
        ])
        sample_y, sample_p = y[sampled], p[sampled]
        sample_pred = (sample_p >= 0.5).astype(int)
        records.append(metric_row(sample_y, sample_pred, sample_p))
    frame = pd.DataFrame(records)
    return {
        metric: (frame[metric].quantile(0.025), frame[metric].quantile(0.975))
        for metric in ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    }


def write_figure(results: pd.DataFrame, output: Path, intervals: pd.DataFrame | None = None) -> None:
    """Create the main figure with bootstrap 95% confidence-interval error bars."""
    order = ["Majority baseline", "Logistic Regression", "Class-weighted Decision Tree", "Random Forest"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    colors = ["#64748b", "#2563eb", "#d97706", "#059669"]
    x = np.arange(len(HORIZONS))
    width = 0.19
    for ax, metric, label in zip(
        axes, ["pr_auc", "recall"], ["Test PR-AUC", "Test recall (bankrupt class)"]
    ):
        for index, (model, color) in enumerate(zip(order, colors)):
            subset = results[results["model"] == model].set_index("horizon").reindex(HORIZONS)
            values = subset[metric].to_numpy()
            yerr = None
            if intervals is not None:
                ci_subset = intervals[
                    (intervals["model"] == model) & (intervals["metric"] == metric)
                ].set_index("horizon").reindex(HORIZONS)
                yerr = np.vstack([
                    values - ci_subset["ci_95_lower"].to_numpy(),
                    ci_subset["ci_95_upper"].to_numpy() - values,
                ])
            ax.bar(
                x + (index - 1.5) * width, values, width, label=model, color=color,
                yerr=yerr, capsize=3, error_kw={"ecolor": "#1f2937", "elinewidth": 0.8},
            )
        ax.set_title(label, fontweight="bold")
        ax.set_xticks(x, [item.replace("year", " year") for item in HORIZONS])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score", fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle(
        "Final test-set performance across bankruptcy prediction horizons\n"
        "Error bars show stratified-bootstrap 95% confidence intervals",
        fontsize=15, fontweight="bold",
    )
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_public_feature_ablation(root: Path, output: Path, selected: pd.DataFrame) -> pd.DataFrame:
    """Compare the selected Random Forest with all ratios versus 10 core ratios.

    This is intentionally validation only: it checks sensitivity to a smaller,
    practical feature set without revisiting the final test data.
    """
    rows = []
    forest_choices = selected[selected["model"] == "Random Forest"].set_index("horizon")
    for horizon in HORIZONS:
        folder = root / "data" / "cleaned" / horizon
        train = pd.read_csv(folder / f"{horizon}_train.csv")
        validation = pd.read_csv(folder / f"{horizon}_validation.csv")
        X_train, y_train = train.drop(columns="bankrupt"), train["bankrupt"]
        X_validation, y_validation = validation.drop(columns="bankrupt"), validation["bankrupt"]
        params = json.loads(forest_choices.loc[horizon, "selected_parameters"])

        for input_set, features in [
            ("All 64 financial ratios", list(X_train.columns)),
            ("10 core public-financial-statement ratios", PUBLIC_FINANCIAL_FEATURES),
        ]:
            model = make_model("Random Forest", params)
            model.fit(X_train[features], y_train)
            probabilities = model.predict_proba(X_validation[features])[:, 1]
            predictions = model.predict(X_validation[features])
            rows.append({
                "horizon": horizon,
                "model": "Random Forest",
                "input_set": input_set,
                "feature_count": len(features),
                "parameters": json.dumps(params, sort_keys=True),
                **metric_row(y_validation, predictions, probabilities),
                "split_used": "validation only; final test set not used",
            })

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "public_feature_ablation.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    labels = ["All 64 ratios", "10 core ratios"]
    colors = ["#059669", "#7c3aed"]
    x = np.arange(len(HORIZONS))
    for axis, metric, title in zip(
        axes, ["pr_auc", "recall"], ["Validation PR-AUC", "Validation recall"]
    ):
        for index, (input_set, label, color) in enumerate(zip(frame["input_set"].unique(), labels, colors)):
            subset = frame[frame["input_set"] == input_set].set_index("horizon").reindex(HORIZONS)
            axis.bar(x + (index - 0.5) * 0.35, subset[metric], 0.35, label=label, color=color)
        axis.set_title(title, fontweight="bold")
        axis.set_xticks(x, [item.replace("year", " year") for item in HORIZONS])
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Score", fontweight="bold")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Feature ablation: full ratios versus a practical 10-ratio subset", fontweight="bold")
    fig.savefig(output / "public_feature_ablation_figure.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return frame


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output = root / "results" / "milestone5"
    output.mkdir(parents=True, exist_ok=True)
    log_rows, tuning_rows, final_rows, ci_rows, error_rows = [], [], [], [], []
    rng = np.random.default_rng(SEED)

    for horizon in HORIZONS:
        folder = root / "data" / "cleaned" / horizon
        train = pd.read_csv(folder / f"{horizon}_train.csv")
        validation = pd.read_csv(folder / f"{horizon}_validation.csv")
        test = pd.read_csv(folder / f"{horizon}_test.csv")
        X_train, y_train = train.drop(columns="bankrupt"), train["bankrupt"]
        X_validation, y_validation = validation.drop(columns="bankrupt"), validation["bankrupt"]
        X_test, y_test = test.drop(columns="bankrupt"), test["bankrupt"]

        for model_name in MODEL_GRIDS:
            selected = tune_one_model(
                model_name, X_train, y_train, X_validation, y_validation, horizon, log_rows
            )
            params, validation_scores = selected["params"], selected["scores"]
            tuning_rows.append({
                "horizon": horizon, "model": model_name,
                "selected_parameters": json.dumps(params, sort_keys=True),
                "validation_pr_auc": validation_scores["pr_auc"],
                "validation_recall": validation_scores["recall"],
                "validation_false_positives": validation_scores["false_positives"],
            })

            # Refit only the selected configuration using train + validation.
            X_fit = pd.concat([X_train, X_validation], ignore_index=True)
            y_fit = pd.concat([y_train, y_validation], ignore_index=True)
            started = time.perf_counter()
            final_model = make_model(model_name, params)
            final_model.fit(X_fit, y_fit)
            test_probabilities = final_model.predict_proba(X_test)[:, 1]
            test_predictions = final_model.predict(X_test)
            scores = metric_row(y_test, test_predictions, test_probabilities)
            elapsed = time.perf_counter() - started
            final_rows.append({
                "horizon": horizon, "model": model_name,
                "selected_parameters": json.dumps(params, sort_keys=True), **scores,
            })
            log_rows.append({
                "run_id": f"{horizon}-{model_name.lower().replace(' ', '-')}-final-test",
                "stage": "final_test",
                "horizon": horizon,
                "model": model_name,
                "parameters": json.dumps(params, sort_keys=True),
                "selection_metric": SELECTION_METRIC,
                "seed": SEED,
                "split_used": "training + validation fit; test evaluated once after selection",
                **{key: scores[key] for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "false_positives", "false_negatives"]},
                "runtime_seconds": round(elapsed, 2),
                "status": "completed",
                "notes": "Final test result; not used to choose configuration.",
            })

            intervals = bootstrap_intervals(y_test, test_probabilities, rng)
            for metric, (lower, upper) in intervals.items():
                ci_rows.append({
                    "horizon": horizon, "model": model_name, "metric": metric,
                    "estimate": scores[metric], "ci_95_lower": lower, "ci_95_upper": upper,
                    "bootstrap_samples": BOOTSTRAP_SAMPLES,
                })

            # Store one anonymous false-negative and false-positive example if present.
            examples = pd.DataFrame({
                "test_row_index": X_test.index,
                "actual_class": y_test.to_numpy(),
                "predicted_class": test_predictions,
                "predicted_bankruptcy_probability": test_probabilities,
            })
            for label, condition in [
                ("false_negative", (examples.actual_class == 1) & (examples.predicted_class == 0)),
                ("false_positive", (examples.actual_class == 0) & (examples.predicted_class == 1)),
            ]:
                found = examples[condition]
                if not found.empty:
                    row = found.iloc[0].to_dict()
                    error_rows.append({"horizon": horizon, "model": model_name, "error_type": label, **row})

    final_results = pd.DataFrame(final_rows)
    tuning_results = pd.DataFrame(tuning_rows)
    final_results.to_csv(output / "main_test_results.csv", index=False)
    tuning_results.to_csv(output / "selected_configurations.csv", index=False)
    pd.DataFrame(log_rows).to_csv(output / "experiment_log.csv", index=False)
    confidence_intervals = pd.DataFrame(ci_rows)
    confidence_intervals.to_csv(output / "bootstrap_confidence_intervals.csv", index=False)
    pd.DataFrame(error_rows).to_csv(output / "error_examples.csv", index=False)
    pd.DataFrame(log_rows).query("stage == 'validation_tuning' and model == 'Class-weighted Decision Tree'").to_csv(
        output / "tree_depth_sensitivity.csv", index=False
    )
    run_public_feature_ablation(root, output, tuning_results)
    write_figure(final_results, output / "main_results_figure.png", confidence_intervals)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": SEED,
        "selection_metric": SELECTION_METRIC,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "horizons": HORIZONS,
        "python": sys.version,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
        "git_commit": subprocess.getoutput(f'git -C "{root}" rev-parse HEAD'),
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Created results package: {output}")


if __name__ == "__main__":
    main()
