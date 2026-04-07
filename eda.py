import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

def distribution(data, export_dir):
    df = data.copy()
    df = df.dropna()

    pdf_path = f'{export_dir}/average_plots_per_opleidingsniveau.pdf'

    numeric_fix = ["sigaretten_per_dag", "cholesterol", "BMI", "glucose"]

    for col in numeric_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    numeric_cols = df.select_dtypes(include='number').columns.drop("Individu-ID", errors="ignore")
    categorical_cols = df.select_dtypes(exclude='number').columns

    with PdfPages(pdf_path) as pdf:

        # Numeric plots
        for col in numeric_cols:
            grouped = df.groupby("opleidingsniveau")[col].mean()

            fig = plt.figure(figsize=(8,4))
            plt.bar(grouped.index.astype(str), grouped.values)
            plt.title(f"Gemiddelde {col} per opleidingsniveau")
            plt.xlabel("Opleidingsniveau")
            plt.ylabel(col)
            plt.xticks(rotation=45)

            pdf.savefig(fig)
            plt.close(fig)

        # Categorical plots
        for col in categorical_cols:
            counts = df.groupby(["opleidingsniveau", col]).size().unstack(fill_value=0)

            fig = plt.figure(figsize=(8,4))

            x = range(len(counts.index))                 # base x positions
            width = 0.8 / len(counts.columns)            # width of each bar

            for i, category in enumerate(counts.columns):
                plt.bar(
                    [p + i * width for p in x],          # shifted positions
                    counts[category],
                    width=width,
                    label=str(category)
                )

            plt.title(f"Verdeling van {col} per opleidingsniveau")
            plt.xlabel("Opleidingsniveau")
            plt.ylabel("Aantal")
            plt.xticks([p + width*(len(counts.columns)/2) for p in x],
                    counts.index.astype(str),
                    rotation=45)
            plt.legend(title=col)

            pdf.savefig(fig)
            plt.close(fig)

    pdf_path = f'{export_dir}/average_plots_per_leeftijd.pdf'

    numeric_fix = ["sigaretten_per_dag", "cholesterol", "BMI", "glucose"]

    for col in numeric_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols = df.select_dtypes(include='number').columns.drop("Individu-ID", errors="ignore")
    categorical_cols = df.select_dtypes(exclude='number').columns

    with PdfPages(pdf_path) as pdf:

        # Numeric plots
        for col in numeric_cols:
            grouped = df.groupby("leeftijd")[col].mean()

            fig = plt.figure(figsize=(8,4))
            plt.bar(grouped.index.astype(str), grouped.values)
            plt.title(f"Gemiddelde {col} per leeftijd")
            plt.xlabel("Leeftijd")
            plt.ylabel(col)
            plt.xticks(rotation=45)

            pdf.savefig(fig)
            plt.close(fig)

        # Categorical plots
        for col in categorical_cols:
            counts = df.groupby(["leeftijd", col]).size().unstack(fill_value=0)

            fig = plt.figure(figsize=(8,4))

            x = range(len(counts.index))                 # base x positions
            width = 0.8 / len(counts.columns)            # width of each bar

            for i, category in enumerate(counts.columns):
                plt.bar(
                    [p + i * width for p in x],          # shifted positions
                    counts[category],
                    width=width,
                    label=str(category)
                )

            plt.title(f"Verdeling van {col} per leeftijd")
            plt.xlabel("Leeftijd")
            plt.ylabel("Aantal")
            plt.xticks(
                [p + width*(len(counts.columns)/2) for p in x],
                counts.index.astype(str),
                rotation=45
            )
            plt.legend(title=col)

            pdf.savefig(fig)
            plt.close(fig)
    return