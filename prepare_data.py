import pandas as pd
from sklearn.model_selection import train_test_split

# Laad de originele dataset
df = pd.read_csv("model/data/data-studenten.csv")

# Split in train/test
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["prognose10jaar"]
)

# Testset mag GEEN targetkolom bevatten
test_features = test_df.drop(columns=["prognose10jaar"])

# Sla bestanden op in de root
train_df.to_csv("train.csv", index=False)
test_features.to_csv("test.csv", index=False)

print("Train/test bestanden zijn aangemaakt!")
