import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

class CHDClassifier:
    def __init__(self):
        # Load training data
        df = pd.read_csv("model/data/data-studenten.csv")

        # Clean missing values ("?" → NaN)
        df = df.replace("?", np.nan)

        # Convert + / - to 1 / 0
        binary_cols = ["hypertensie", "hartinfarct", "diabetes", "nierziekte"]
        for col in binary_cols:
            df[col] = df[col].map({"+": 1, "-": 0})

        # Convert numeric columns
        numeric_cols = [
            "leeftijd","sigaretten_per_dag","slaapscore","cholesterol",
            "bovendruk","onderdruk","BMI","hartslag","glucose"
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows with missing target
        df = df.dropna(subset=["prognose10jaar"])

        # Encode target
        y = df["prognose10jaar"].map({"CHD-": 0, "CHD+": 1}).values

        # Drop non-feature columns
        X = df.drop(columns=["prognose10jaar", "Individu-ID"]).copy()

        # Encode geslacht (M/V)
        X["geslacht"] = X["geslacht"].map({"M": 1, "V": 0})

        # Encode opleidingsniveau (1–4)
        X["opleidingsniveau"] = pd.to_numeric(X["opleidingsniveau"], errors="coerce")

        # Fill remaining missing values
        X = X.fillna(X.median())

        # Train/test split for model selection
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Define both models
        rf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=3,
    class_weight={0:1, 1:5},
    random_state=42
)
)
        ])

        et = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42
            ))
        ])

        gb = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_features="sqrt",
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42
            ))
        ])

        rf.fit(X_train, y_train)
        et.fit(X_train, y_train)
        gb.fit(X_train, y_train)

        rf_acc = accuracy_score(y_test, rf.predict(X_test))
        et_acc = accuracy_score(y_test, et.predict(X_test))
        gb_acc = accuracy_score(y_test, gb.predict(X_test))

        # Kies beste model
        best_model = rf
        best_acc = rf_acc

        if et_acc > best_acc:
            best_model = et
            best_acc = et_acc

        if gb_acc > best_acc:
            best_model = gb
            best_acc = gb_acc

        self.model = best_model

        # Maak één PDF voor alle evaluaties
        with PdfPages("model_evaluation.pdf") as pdf:
            # MODELVERGELIJKING
            fig = plt.figure(figsize=(8, 4))
            plt.title("Modelvergelijking (Accuracy)")

            models = ["RandomForest", "ExtraTrees", "GradientBoosting"]
            scores = [rf_acc, et_acc, gb_acc]

            plt.bar(models, scores, color=["#4c72b0", "#55a868", "#c44e52"])
            plt.ylabel("Accuracy")
            plt.ylim(0, 1)

            for i, v in enumerate(scores):
                plt.text(i, v + 0.01, f"{v:.3f}", ha="center")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # MODELVERGELIJKING (precision, recall, F1)
            # Bereken scores voor alle modellen
            rf_pred = rf.predict(X_test)
            et_pred = et.predict(X_test)
            gb_pred = gb.predict(X_test)

            model_names = ["RandomForest", "ExtraTrees", "GradientBoosting"]
            precisions = [
                precision_score(y_test, rf_pred),
                precision_score(y_test, et_pred),
                precision_score(y_test, gb_pred)
            ]
            recalls = [
                recall_score(y_test, rf_pred),
                recall_score(y_test, et_pred),
                recall_score(y_test, gb_pred)
            ]
            f1s = [
                f1_score(y_test, rf_pred),
                f1_score(y_test, et_pred),
                f1_score(y_test, gb_pred)
            ]

            # Maak een tabelpagina
            fig = plt.figure(figsize=(8, 4))
            plt.title("Modelvergelijking (Precision, Recall, F1)", fontsize=14)

            table_data = [
                ["Model", "Precision", "Recall", "F1-score"],
                [model_names[0], f"{precisions[0]:.3f}", f"{recalls[0]:.3f}", f"{f1s[0]:.3f}"],
                [model_names[1], f"{precisions[1]:.3f}", f"{recalls[1]:.3f}", f"{f1s[1]:.3f}"],
                [model_names[2], f"{precisions[2]:.3f}", f"{recalls[2]:.3f}", f"{f1s[2]:.3f}"],
            ]

            plt.axis("off")
            table = plt.table(
                cellText=table_data,
                loc="center",
                cellLoc="center"
            )
            table.scale(1, 2)
            pdf.savefig(fig)
            plt.close(fig)

            #  CONFUSION MATRIX
            # CONFUSION MATRIX (met threshold-tuning)
            y_prob = best_model.predict_proba(X_test)[:, 1]
            y_pred_thresh = (y_prob >= 0.35).astype(int)

            cm = confusion_matrix(y_test, y_pred_thresh)

            fig = plt.figure(figsize=(6, 4))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title("Confusion Matrix (Threshold = 0.35)")
            plt.colorbar()

            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ["Pred 0", "Pred 1"])
            plt.yticks(tick_marks, ["Actual 0", "Actual 1"])

            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                             ha="center", va="center",
                             color="white" if cm[i, j] > thresh else "black")

            plt.ylabel("Actual")
            plt.xlabel("Predicted")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            #  ROC CURVE
            if hasattr(best_model.named_steps["clf"], "predict_proba"):
                # ROC CURVE (met threshold-markering)
                if hasattr(best_model.named_steps["clf"], "predict_proba"):
                    y_prob = best_model.predict_proba(X_test)[:, 1]

                    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)

                    fig = plt.figure(figsize=(6, 4))
                    plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.3f}")
                    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")

                    # Markeer jouw threshold op de ROC-curve
                    # Zoek de dichtstbijzijnde threshold-index
                    idx = (np.abs(thresholds - 0.35)).argmin()
                    plt.scatter(fpr[idx], tpr[idx], color="red", label="Threshold = 0.35")

                    plt.title("ROC Curve (met threshold-markering)")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.legend(loc="lower right")
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

            # FEATURE IMPORTANCE (alleen voor tree‑based models)
            clf = best_model.named_steps["clf"]
            if hasattr(clf, "feature_importances_"):
                importances = clf.feature_importances_
                indices = np.argsort(importances)[::-1]
                feature_names = X.columns

                fig = plt.figure(figsize=(8, 6))
                plt.bar(range(len(importances)), importances[indices])
                plt.xticks(range(len(importances)),
                           feature_names[indices],
                           rotation=90)
                plt.title("Feature Importances")
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            # CLASSIFICATION REPORT (als tekstpagina)
            y_pred_thresh = (y_prob >= 0.35).astype(int)
            report = classification_report(y_test, y_pred_thresh)

            fig = plt.figure(figsize=(8, 6))
            plt.text(0.01, 0.05, report, fontsize=10, family="monospace")
            plt.title("Classification Report")
            plt.axis("off")
            pdf.savefig(fig)
            plt.close(fig)

        # Store training columns for predict()
        self.columns = X.columns

    def predict(self, filename):
        df = pd.read_csv(filename)

        # Zelfde preprocessing als training
        df = df.replace("?", np.nan)

        # Encode geslacht
        df["geslacht"] = df["geslacht"].map({"M": 1, "V": 0})

        # Encode opleidingsniveau
        df["opleidingsniveau"] = pd.to_numeric(df["opleidingsniveau"], errors="coerce")

        # Encode + / - kolommen
        binary_cols = ["hypertensie", "hartinfarct", "diabetes", "nierziekte"]
        for col in binary_cols:
            df[col] = df[col].map({"+": 1, "-": 0})

        # Converteer numerieke kolommen
        numeric_cols = [
            "leeftijd", "sigaretten_per_dag", "slaapscore", "cholesterol",
            "bovendruk", "onderdruk", "BMI", "hartslag", "glucose"
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # eerst kolommen selecteren
        df = df[self.columns]

        # Daarna pas median() doen
        df = df.fillna(df.median())

        return self.model.predict(df)

