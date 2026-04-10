from model import model_factory
import pandas as pd

print("start")

model = model_factory()
print("model gemaakt")

preds = model.predict("test.csv")
print("voorspellingen gemaakt")

print(preds)
print(len(preds))

pd.DataFrame({"prediction": preds}).to_csv("submission.csv", index=False)
print("submission.csv opgeslagen")