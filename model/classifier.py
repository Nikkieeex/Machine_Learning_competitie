import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class CHDClassifier:
    def __init__(self):
        df = pd.read_csv("model/data/data-studenten.csv")
        df = self.clean_dataframe(df, training=True)

        y = df["prognose10jaar"].map({"CHD-": 0, "CHD+": 1}).values

        drop_cols = ["prognose10jaar"]
        if "Individu-ID" in df.columns:
            drop_cols.append("Individu-ID")
        if "Unnamed: 0" in df.columns:
            drop_cols.append("Unnamed: 0")

        X = df.drop(columns=drop_cols).copy()
        self.columns = X.columns

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        X_train_bal, y_train_bal = self.balance_training_data(X_train, y_train)

        balanced_train_df = X_train_bal.copy()
        balanced_train_df["prognose10jaar"] = np.where(y_train_bal == 1, "CHD+", "CHD-")
        balanced_train_df.to_csv("balanced_train.csv", index=False)

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", BaggingClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=5,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42
                ),
                n_estimators=300,
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            ))
        ])

        self.model.fit(X_train_bal, y_train_bal)

        y_prob = self.predict_proba_safe(X_test)

        self.best_threshold, best_metrics = self.find_best_threshold(y_test, y_prob)

        default_preds = (y_prob >= 0.50).astype(int)
        tuned_preds = (y_prob >= self.best_threshold).astype(int)

        default_metrics = self.get_scores(y_test, default_preds)

        print("\n=== BALANCED BAGGING THRESHOLD OVERZICHT ===")
        print(f"Threshold 0.50 -> Accuracy: {default_metrics['accuracy']:.3f}, Precision: {default_metrics['precision']:.3f}, Recall: {default_metrics['recall']:.3f}, F1: {default_metrics['f1']:.3f}")
        print(f"Threshold {self.best_threshold:.2f} -> Accuracy: {best_metrics['accuracy']:.3f}, Precision: {best_metrics['precision']:.3f}, Recall: {best_metrics['recall']:.3f}, F1: {best_metrics['f1']:.3f}")
        print(f"Gekozen threshold: {self.best_threshold:.2f}")

        self.make_pdf_report(y_test, y_prob, tuned_preds, best_metrics)

    def clean_dataframe(self, df, training=False):
        df = df.copy()

        df = df.replace("?", np.nan)

        binary_cols = ["hypertensie", "hartinfarct", "diabetes", "nierziekte"]
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map({"+": 1, "-": 0})

        if "geslacht" in df.columns:
            df["geslacht"] = df["geslacht"].map({"M": 1, "V": 0})

        if "opleidingsniveau" in df.columns:
            df["opleidingsniveau"] = pd.to_numeric(df["opleidingsniveau"], errors="coerce")

        numeric_cols = [
            "leeftijd", "sigaretten_per_dag", "slaapscore", "cholesterol",
            "bovendruk", "onderdruk", "BMI", "hartslag", "glucose"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if training:
            df = df.dropna(subset=["prognose10jaar"])

        for col in ["Unnamed: 0", "Individu-ID"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        df = df.fillna(df.median(numeric_only=True))

        return df

    def balance_training_data(self, X, y):
        train_df = X.copy()
        train_df["target"] = y

        majority = train_df[train_df["target"] == 0]
        minority = train_df[train_df["target"] == 1]

        minority_upsampled = minority.sample(
            n=len(majority),
            replace=True,
            random_state=42
        )

        balanced_df = pd.concat([majority, minority_upsampled], axis=0)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

        y_balanced = balanced_df["target"].values
        X_balanced = balanced_df.drop(columns=["target"])

        return X_balanced, y_balanced

    def predict_proba_safe(self, X):
        if hasattr(self.model.named_steps["clf"], "predict_proba"):
            return self.model.predict_proba(X)[:, 1]

        preds = self.model.predict(X)
        return preds.astype(float)

    def get_scores(self, y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }

    def find_best_threshold(self, y_true, y_prob):
        best_threshold = 0.50
        best_metrics = None
        best_f1 = -1

        for threshold in np.arange(0.10, 0.91, 0.01):
            preds = (y_prob >= threshold).astype(int)
            metrics = self.get_scores(y_true, preds)

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_threshold = threshold
                best_metrics = metrics

        return best_threshold, best_metrics

    def make_pdf_report(self, y_true, y_prob, preds, metrics):
        with PdfPages("model_evaluation.pdf") as pdf:
            fig = plt.figure(figsize=(8, 4))
            plt.title("BalancedBagging metrics")
            names = ["Accuracy", "Precision", "Recall", "F1"]
            values = [
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1"]
            ]
            plt.bar(names, values)
            plt.ylim(0, 1)

            for i, v in enumerate(values):
                plt.text(i, v + 0.01, f"{v:.3f}", ha="center")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            cm = confusion_matrix(y_true, preds)

            fig = plt.figure(figsize=(6, 4))
            plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix (threshold={self.best_threshold:.2f})")
            plt.colorbar()

            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ["Pred 0", "Pred 1"])
            plt.yticks(tick_marks, ["Actual 0", "Actual 1"])

            thresh = cm.max() / 2
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(
                        j,
                        i,
                        format(cm[i, j], "d"),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > thresh else "black"
                    )

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)

            fig = plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], lw=1, linestyle="--")

            idx = (np.abs(thresholds - self.best_threshold)).argmin()
            plt.scatter(fpr[idx], tpr[idx], label=f"Threshold = {self.best_threshold:.2f}")

            plt.title("ROC Curve - BalancedBagging")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            report = classification_report(y_true, preds, zero_division=0)

            fig = plt.figure(figsize=(8, 6))
            plt.text(0.01, 0.05, report, fontsize=10, family="monospace")
            plt.title("Classification Report")
            plt.axis("off")
            pdf.savefig(fig)
            plt.close(fig)

    def predict(self, filename):
        df = pd.read_csv(filename)
        df = self.clean_dataframe(df, training=False)

        df = df[self.columns]

        y_prob = self.predict_proba_safe(df)
        preds = (y_prob >= self.best_threshold).astype(int)

        return np.where(preds == 1, "CHD+", "CHD-")