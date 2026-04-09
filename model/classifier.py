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
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

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

        lr = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight={0: 1, 1: 6},
                max_iter=500,
                solver="lbfgs"
            ))
        ])

        svm = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(
                class_weight={0: 1, 1: 6},
                max_iter=5000
            ))
        ])

        bb = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", BaggingClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=5,
                    class_weight="balanced"
                ),
                n_estimators=200,
                bootstrap=True,
                random_state=42
            ))
        ])

        rf.fit(X_train, y_train)
        et.fit(X_train, y_train)
        gb.fit(X_train, y_train)
        lr.fit(X_train, y_train)
        svm.fit(X_train, y_train)
        bb.fit(X_train, y_train)

        rf_acc = accuracy_score(y_test, rf.predict(X_test))
        et_acc = accuracy_score(y_test, et.predict(X_test))
        gb_acc = accuracy_score(y_test, gb.predict(X_test))
        lr_acc = accuracy_score(y_test,lr.predict(X_test))
        svm_acc = accuracy_score(y_test, svm.predict(X_test))
        bb_acc = accuracy_score(y_test, bb.predict(X_test))

        # Kies beste model
        best_model = rf
        best_acc = rf_acc

        if et_acc > best_acc:
            best_model = et
            best_acc = et_acc

        if gb_acc > best_acc:
            best_model = gb
            best_acc = gb_acc

        if lr_acc > best_acc:
            best_model = lr
            best_acc = lr_acc

        if svm_acc > best_acc:
            best_model = svm
            best_acc = svm_acc

        if bb_acc > best_acc:
            best_model = bb
            best_acc = bb_acc

        self.model = best_model

        # Maak één PDF voor alle evaluaties
        with PdfPages("model_evaluation.pdf") as pdf:
            # MODELVERGELIJKING
            fig = plt.figure(figsize=(8, 4))
            plt.title("Modelvergelijking (Accuracy)")

            models = ["RandomForest", "ExtraTrees", "GradientBoosting", "LogisticRegression", "LinearSVC", "Balanced Bagging"]
            scores = [rf_acc, et_acc, gb_acc, lr_acc, svm_acc, bb_acc]

            plt.bar(models, scores, color=["#4c72b0", "#55a868", "#c44e52", "#8172b3", "#ccb974", "#CCB902"])
            plt.ylabel("Accuracy")
            plt.ylim(0, 1)

            for i, v in enumerate(scores):
                plt.text(i, v + 0.01, f"{v:.3f}", ha="center")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # MODELVERGELIJKING (precision, recall, F1)
            rf_pred = rf.predict(X_test)
            et_pred = et.predict(X_test)
            gb_pred = gb.predict(X_test)
            lr_pred = lr.predict(X_test)
            svm_pred = svm.predict(X_test)
            bb_pred = bb.predict(X_test)

            model_names = ["RandomForest", "ExtraTrees", "GradientBoosting", "LogisticRegression", "LinearSVC", "Balanced Bagging"]
            precisions = [
                precision_score(y_test, rf_pred),
                precision_score(y_test, et_pred),
                precision_score(y_test, gb_pred),
                precision_score(y_test, lr_pred),
                precision_score(y_test, svm_pred),
                precision_score(y_test, bb_pred)
            ]
            recalls = [
                recall_score(y_test, rf_pred),
                recall_score(y_test, et_pred),
                recall_score(y_test, gb_pred),
                recall_score(y_test, lr_pred),
                recall_score(y_test, svm_pred),
                recall_score(y_test, bb_pred)
            ]
            f1s = [
                f1_score(y_test, rf_pred),
                f1_score(y_test, et_pred),
                f1_score(y_test, gb_pred),
                f1_score(y_test, lr_pred),
                f1_score(y_test, svm_pred),
                f1_score(y_test, bb_pred)
            ]

            # Maak een tabelpagina
            fig = plt.figure(figsize=(8, 4))
            plt.title("Modelvergelijking (Precision, Recall, F1)", fontsize=14)

            table_data = [
                ["Model", "Precision", "Recall", "F1-score"],
                *[
                    [model_names[i], f"{precisions[i]:.3f}", f"{recalls[i]:.3f}", f"{f1s[i]:.3f}"]
                    for i in range(len(model_names))
                ]
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
            if hasattr(best_model.named_steps["clf"], "predict_proba"):
                y_prob = best_model.predict_proba(X_test)[:, 1]
            else:
                # SVM fallback via decision_function
                decision = best_model.decision_function(X_test)
                y_prob = (decision - decision.min()) / (decision.max() - decision.min())

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

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            #  ROC CURVE
            if hasattr(best_model.named_steps["clf"], "predict_proba"):
                y_prob = best_model.predict_proba(X_test)[:, 1]

                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)

                fig = plt.figure(figsize=(6, 4))
                plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.3f}")
                plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")

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

            # COEFFICIENT PLOT (Logistic Regression)
            clf_lr = lr.named_steps["clf"]

            if hasattr(clf_lr, "coef_"):
                coefs = clf_lr.coef_[0]
                feature_names = X.columns
                indices = np.argsort(np.abs(coefs))[::-1]  # sorteer op absolute impact

                fig = plt.figure(figsize=(8, 6))
                plt.bar(range(len(coefs)), coefs[indices], color="purple")
                plt.xticks(range(len(coefs)), feature_names[indices], rotation=90)
                plt.title("Logistic Regression Coefficients")
                plt.ylabel("Coefficient Value")
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            # CALIBRATION CURVE (Logistic Regression)
            y_prob_lr = lr.predict_proba(X_test)[:, 1]
            prob_true, prob_pred = calibration_curve(y_test, y_prob_lr, n_bins=10)

            fig = plt.figure(figsize=(6, 4))
            plt.plot(prob_pred, prob_true, marker="o", color="purple", label="Logistic Regression")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")

            plt.title("Calibration Curve - Logistic Regression")
            plt.xlabel("Gemiddelde voorspelde kans")
            plt.ylabel("Werkelijke kans")
            plt.legend(loc="upper left")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # CLASSIFICATION REPORT
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

